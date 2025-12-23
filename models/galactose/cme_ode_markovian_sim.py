'''
CME-ODE Simulation with Markovian Promoter Dynamics
Author: Tianyu Wu, 2024

This version replaces the CME-based promoter binding reactions with
explicit Markovian state dynamics computed in the hook.

Key differences from the original cme_ode_sim.py:
1. Promoter binding/unbinding reactions are NOT added to CME
2. The hook uses a Markovian model to evolve promoter state probabilities
3. Gene states (empty/active/repressed) are stochastically sampled and 
   written back to CME species
4. Activation statistics are tracked throughout the simulation
'''

'''
Part1: Package Import
'''
from jLM import CME
from jLM.units import *
from jLM import LMLogger
import logging
LMLogger.setLMLoggerLevel(logging.INFO)

import sys
import os
from contextlib import redirect_stdout
import numpy as np
import scipy
import scipy.optimize
import copy
import json

'''
Part2: Allows user to input system conditions
'''
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('-gex', '--GAE', required=True, help='External galactose concentration (mM)')
ap.add_argument('-gic', '--GAI', required=True, help='Internal galactose concentration (mM)')
ap.add_argument('-rep', '--replicates', required=True, help='Number of replicates')
ap.add_argument('-delt', '--delt', required=True, help='Communication timestep (seconds)')
ap.add_argument('-t', '--simTime', type=int, default=750, help='Simulation time (default: 750 min)')
ap.add_argument('-bscale', '--binding_scale', type=float, default=1.0, 
                help='Binding kinetics scale (1.0=normal, >1=faster equilibration)')
ap.add_argument('--csv_out', action='store_true', help='Export trajectory to CSV')

args = ap.parse_args()

import datetime

save_path = "output_markovian/" + datetime.datetime.now().strftime("%d%m%Y") + "/"
output_file = f"gal_cme_markov_gae{args.GAE}_gai{args.GAI}_rep{args.replicates}_delta{args.delt}_time{args.simTime}.lm"
log_file = f"log_cme_markov_gae{args.GAE}_gai{args.GAI}_rep{args.replicates}_delta{args.delt}_time{args.simTime}.log"
stats_file = f"activation_stats_gae{args.GAE}_gai{args.GAI}_rep{args.replicates}.json"

print("pid for this program is:", os.getpid())

'''=============================================================
                    Main Code starts from here
=============================================================='''

'''=============================================================
part3: CME-ODE solver with Markovian Promoter Dynamics
=============================================================='''

from scipy.integrate import odeint 
import ode_func as ode_solver
import lm
from markovian_promoter import MarkovianPromoterModel, PromoterStateTracker


class CMEODEMarkovianSolver(lm.GillespieDSolver):
    """
    CME-ODE solver with Markovian promoter dynamics.
    
    In each hook call:
    1. Get G4d and G80d concentrations from CME
    2. Evolve Markovian promoter model
    3. Sample discrete gene states
    4. Update CME species to reflect new gene states
    5. Track activation statistics
    """
    
    def initializeSolver(self, counts, delt, rxnsICareAbout, ks, gae, ode_step, 
                         speciesCount, binding_scale=1.0):
        """Initialize the solver with Markovian promoter model."""
        
        # Save initial conditions
        self.ic = (counts, delt, rxnsICareAbout, ks, gae, ode_step, speciesCount, binding_scale)
        
        # Initialize Markovian promoter model
        self.promoter_model = MarkovianPromoterModel(binding_scale=binding_scale)
        
        # State tracker for statistics
        self.tracker = PromoterStateTracker(list(self.promoter_model.PROMOTERS.keys()))
        
        # Storage for all replicates' statistics
        self.all_replicate_stats = []
        self.current_replicate = 0
        
        # Set initial conditions
        self.restart()
    
    def restart(self):
        """Reset for a new replicate."""
        self.oldtime = 0
        self.counts = copy.deepcopy(self.ic[0])
        self.delt = copy.deepcopy(self.ic[1])
        self.rxnsICareAbout = copy.deepcopy(self.ic[2])
        self.ks = copy.deepcopy(self.ic[3])
        self.gae = copy.deepcopy(self.ic[4])
        self.odestep = copy.deepcopy(self.ic[5])
        self.species = copy.deepcopy(self.ic[6])
        
        # Reset Markovian model and tracker
        self.promoter_model.reset()
        self.tracker.reset()
    
    def hookSimulation(self, time):
        """
        Hook called at each communication timestep.
        
        This is where we:
        1. Run the ODE solver for galactose transport
        2. Evolve the Markovian promoter model
        3. Update gene states in CME
        """
        if time == 0.0:
            # New replicate starting
            if self.current_replicate > 0:
                # Save statistics from previous replicate
                stats = self.tracker.get_summary()
                self.all_replicate_stats.append({
                    'replicate': self.current_replicate,
                    'stats': stats
                })
                print(f"\n=== Replicate {self.current_replicate} Activation Statistics ===")
                for gene, s in stats.items():
                    print(f"  {gene}: {s['cumulative_active_fraction']*100:.1f}% active")
            
            self.current_replicate += 1
            print(f"\nNew Replicate {self.current_replicate}", flush=True)
            self.restart()
            return 0
        
        # Update species counts from CME
        self.species.update(self)
        
        # =========================================================
        # PART 1: ODE for galactose transport (same as before)
        # =========================================================
        rates = np.zeros(len(self.rxnsICareAbout))
        self.counts[0] = self.species['GAI']
        self.counts[1] = self.species['G2GAI']
        self.counts[2] = self.species['G2GAE']
        self.counts[3] = self.species['G1GAI']
        self.counts[4] = self.species['G1']
        self.counts[5] = self.species['G2']
        
        # ODE solver
        if time < 100:
            stepsize = self.odestep / 10
            sol = odeint(ode_solver.dxdt, self.counts, 
                        np.linspace(time, time + self.delt, int(np.ceil(self.delt/stepsize)) + 1),
                        args=(rates, self.gae), hmax=stepsize)
        else:
            sol = odeint(ode_solver.dxdt, self.counts,
                        np.linspace(time, time + self.delt, int(np.ceil(self.delt/self.odestep)) + 1),
                        args=(rates, self.gae), hmax=self.odestep)
        
        self.counts = sol[-1]
        
        # Update galactose-related species
        totalG2 = self.species['G2GAI'] + self.species['G2GAE'] + self.species['G2']
        self.species['G2GAI'] = round(self.counts[1])
        self.species['G2GAE'] = round(self.counts[2])
        self.species['G2'] = round(totalG2 - self.species['G2GAI'] - self.species['G2GAE'])
        
        totalG1 = self.species['G1GAI'] + self.species['G1']
        self.species['G1GAI'] = round(self.counts[3])
        self.species['G1'] = round(totalG1 - self.species['G1GAI'])
        
        self.species['GAI'] = round(self.counts[0])
        
        # =========================================================
        # PART 2: Markovian Promoter Dynamics
        # =========================================================
        
        # Get activator and repressor concentrations from CME
        G4d = self.species['G4d']  # Gal4p dimer
        G80d = self.species['G80d']  # Gal80p dimer (nuclear)
        
        # Evolve Markovian model
        dt = self.delt  # time step in minutes
        gene_states = self.promoter_model.step(dt, G4d, G80d)
        
        # Get active probabilities for tracking
        active_probs = {}
        for gene in self.promoter_model.PROMOTERS:
            active_probs[gene] = self.promoter_model.get_active_fraction(gene, G4d, G80d)
        
        # Record for tracking
        self.tracker.record(time, gene_states, active_probs, dt)
        
        # =========================================================
        # PART 3: Update CME gene species based on sampled states
        # =========================================================
        
        for gene_name, config in self.promoter_model.PROMOTERS.items():
            state = gene_states[gene_name]
            
            # Get current CME species counts for this gene
            empty_species = config['empty']
            active_species = config['active']
            repressed_species = config['repressed']
            
            current_empty = self.species[empty_species]
            current_active = self.species[active_species]
            current_repressed = self.species[repressed_species]
            
            # Total should be 1 (single copy gene)
            total = current_empty + current_active + current_repressed
            
            # Set new state (one copy of gene, in the sampled state)
            if state == 'empty':
                self.species[empty_species] = total
                self.species[active_species] = 0
                self.species[repressed_species] = 0
            elif state == 'active':
                self.species[empty_species] = 0
                self.species[active_species] = total
                self.species[repressed_species] = 0
            else:  # repressed
                self.species[empty_species] = 0
                self.species[active_species] = 0
                self.species[repressed_species] = total
        
        self.oldtime = time
        return 1
    
    def get_all_statistics(self):
        """Get statistics from all replicates."""
        # Add current replicate if not already added
        if self.tracker.time_points:
            stats = self.tracker.get_summary()
            self.all_replicate_stats.append({
                'replicate': self.current_replicate,
                'stats': stats
            })
        return self.all_replicate_stats


# Instantiate the solver
odeHookSolver = CMEODEMarkovianSolver()

'''=============================================================
part4: Species definition (same as before, but promoter reactions handled in hook)
=============================================================='''

# ODE species
ode_species = ['GAI', 'G2GAI', 'G2GAE', 'G1GAI', 'G1', 'G2']

# CME species (same as before)
cme_species = ['R1', 'R2', 'R3', 'R4', 'reporter_rna', 'R80', 'G1', 'G2', 'G3', 'G3i', 
               'G4', 'G4d', 'reporter', 'G80', 'G80C', 'G80d', 'G80Cd', 'G80G3i', 'GAI',
               'DG1', 'DG1_G4d', 'DG1_G4d_G80d', 
               'DG2', 'DG2_G4d', 'DG2_G4d_G80d',
               'DG3', 'DG3_G4d', 'DG3_G4d_G80d',
               'DGrep', 'DGrep_G4d', 'DGrep_G4d_G80d',
               'DG80', 'DG80_G4d', 'DG80_G4d_G80d',
               'G2GAI', 'G2GAE', 'G1GAI']

# Simulation time
simTime = int(args.simTime)  # min
delt = float(args.delt) / 60.0  # Convert to minutes

# Create simulation object
sim = CME.CMESimulation()

# Create species count object
import species_counts
mySpecies = species_counts.SpeciesCounts(sim)

# Define species
sim.defineSpecies(cme_species)

# Reactions for ODE
Frxns = []
init_ks = []

# External galactose
Gae = float(args.GAE) / (4.65e-8)  # Convert to molecules

'''=============================================================
Part5: Add reactions for CME (WITHOUT promoter binding reactions)
=============================================================='''
from cme_rxns import transcription, translation, dimerization, g80region_swap, g3_rxns

# Transcription reactions (depend on promoter states set by hook)
transcription.getTranscriptionReactions(sim)

# Translation reactions
translation.getTranslationReactions(sim)

# NOTE: We DO NOT add promoter binding reactions here!
# They are handled by the Markovian model in the hook
# regulators_promoters.getDNAPromoterReactions(sim)  # REMOVED

# Dimerization reactions
dimerization.getDimerizationReactions(sim)

# Gal80 transport reactions
g80region_swap.getG80TransportReactions(sim)

# G3 reactions (including Gal3-Gal80 binding)
g3_rxns.getG3Reactions(sim, Frxns, init_ks)


'''=============================================================
Part 6: Set initial conditions
=============================================================='''

def setInitialCounts(sim, cme_species, gai):
    gai_molec = float(gai) / (4.65e-8)
    
    # Initial counts (same as original)
    # Genes start in empty state (DGx = 1, DGx_G4d = 0, DGx_G4d_G80d = 0)
    cme_count_list = [
        1, 1, 1, 1, 1,  # R1, R2, R3, R4, reporter_rna
        1.18715948592467,  # R80
        132.318563460887,  # G1
        1156.91017704601,  # G2
        4341.70321120979,  # G3
        0,  # G3i
        1,  # G4
        308.921734355756,  # G4d
        132.317774287091,  # reporter
        1,  # G80
        1,  # G80C
        157.246650776274,  # G80d
        157.239961338382,  # G80Cd
        0,  # G80G3i
        gai_molec,  # GAI
        1.0, 0.0, 0.0,  # DG1, DG1_G4d, DG1_G4d_G80d (start empty)
        1.0, 0.0, 0.0,  # DG2, DG2_G4d, DG2_G4d_G80d
        1.0, 0.0, 0.0,  # DG3, DG3_G4d, DG3_G4d_G80d
        1.0, 0.0, 0.0,  # DGrep, DGrep_G4d, DGrep_G4d_G80d
        1.0, 0.0, 0.0,  # DG80, DG80_G4d, DG80_G4d_G80d
        0.0, 0.0, 0.0   # G2GAI, G2GAE, G1GAI
    ]
    
    ode_count_list = [gai_molec, 0, 0, 0, 132.318563460887, 1156.91017704601]
    
    # Add particles
    for i in range(len(cme_species)):
        sim.addParticles(species=cme_species[i], count=int(round(cme_count_list[i])))
    
    return ode_count_list

ode_counts = setInitialCounts(sim, cme_species, args.GAI)


'''=============================================================
Part 6b: Set up simulation
=============================================================='''

try:
    sim.setHookInterval(delt)
    sim.setWriteInterval(1.0)
except AttributeError:
    sim.setWriteInterval(delt)

sim.setSimulationTime(args.simTime)

# Output file
my_lm_file = save_path + output_file

# Create output directory
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Remove existing file
if os.path.exists(my_lm_file):
    os.remove(my_lm_file)

sim.save(my_lm_file)

# ODE step size
odestep = 0.0001


'''=============================================================
Part 7: Run the simulation
=============================================================='''

with open(str(save_path + log_file), 'w') as f, redirect_stdout(f):
    
    # Initialize solver with Markovian model
    odeHookSolver.initializeSolver(
        ode_counts, delt, Frxns, np.asarray(init_ks), Gae, odestep, mySpecies,
        binding_scale=float(args.binding_scale)
    )
    
    # Run simulation
    sim.runSolver(filename=my_lm_file, solver=odeHookSolver, replicates=int(args.replicates))
    
    # Print final statistics
    print("\n" + "=" * 60)
    print("FINAL ACTIVATION STATISTICS (All Replicates)")
    print("=" * 60)
    
    all_stats = odeHookSolver.get_all_statistics()
    
    # Compute averages across replicates
    avg_activation = {gene: [] for gene in MarkovianPromoterModel.PROMOTERS.keys()}
    
    for rep_data in all_stats:
        for gene, stats in rep_data['stats'].items():
            avg_activation[gene].append(stats['cumulative_active_fraction'])
    
    print("\nGene Activation Fractions:")
    print("-" * 40)
    for gene in avg_activation:
        if avg_activation[gene]:
            mean = np.mean(avg_activation[gene])
            std = np.std(avg_activation[gene])
            print(f"  {gene:10}: {mean*100:.2f}% ± {std*100:.2f}%")
    
    # Save statistics to JSON
    with open(save_path + stats_file, 'w') as stats_f:
        json.dump({
            'parameters': {
                'GAE': args.GAE,
                'GAI': args.GAI,
                'replicates': args.replicates,
                'delt': args.delt,
                'simTime': args.simTime,
                'binding_scale': args.binding_scale
            },
            'replicate_stats': all_stats,
            'average_activation': {gene: {
                'mean': float(np.mean(vals)) if vals else 0,
                'std': float(np.std(vals)) if vals else 0
            } for gene, vals in avg_activation.items()}
        }, stats_f, indent=2)
    
    print(f"\nStatistics saved to: {save_path + stats_file}")
    
    f.close()

sys.stdout.flush()
print(f"\nSimulation complete. Output: {my_lm_file}")
print(f"Statistics: {save_path + stats_file}")

# Export CSV if requested
if args.csv_out:
    print("\nExporting CSV...")
    try:
        import h5py
        # Open the .lm file (HDF5 format)
        with h5py.File(my_lm_file, 'r') as f:
            # Extract time
            # Path depends on solver version, but usually Simulations/0000001/SpeciesCountTimes
            sim_time = f['Simulations/0000001/SpeciesCountTimes'][()]
            
            # Reshape species counts: HDF5 is (n_timepoints, n_species)
            counts = f['Simulations/0000001/SpeciesCounts'][()]
            
            # Species names
            species_names_ds = f['Parameters/SpeciesNames'][()]
            species_names = []
            for s in species_names_ds:
                if hasattr(s, 'decode'):
                    species_names.append(s.decode('utf-8'))
                else:
                    species_names.append(str(s))
            
            # Create DataFrame
            import pandas as pd
            df = pd.DataFrame(counts, columns=species_names)
            
            # Ensure time matches length
            if len(sim_time) == len(df):
                df.insert(0, 'Time', sim_time)
            else:
                print(f"Warning: Time length ({len(sim_time)}) != Counts length ({len(df)})")
                # Try to fix by truncation or padding? Usually they match.
                # Ifcounts is (n, t), then SpeciesCountTimes should be (t,)
                pass
            
            # Save to CSV
            csv_file = save_path + output_file.replace('.lm', '.csv')
            df.to_csv(csv_file, index=False)
            print(f"CSV exported to: {csv_file}")
            
    except Exception as e:
        print(f"Error exporting CSV: {e}")
        # Fallback if h5py/pandas issue?
        pass



