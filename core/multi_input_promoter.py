"""
Multi-Input Markovian Promoter Model

This module extends the Markovian promoter framework to support promoters
regulated by multiple transcription factors (TFs) with combinatorial logic.

Classes:
    MultiInputPromoter: A promoter model supporting multiple TF inputs and custom logic.
"""

import numpy as np
import itertools
from .markovian_promoter import MarkovianPromoterModel

class MultiInputPromoter:
    """
    Simulates a promoter regulated by multiple Transcription Factors (TFs).
    
    This class manages the discrete states of a promoter that can be bound by
    different TFs. It constructs a state space representing all possible
    combinations of bound TFs and transitions between them based on individual
    binding/unbinding rates.
    
    Attributes:
        tf_configs (dict): Configuration for each TF input.
            Format: {
                'TF_name': {
                    'k_on': float,  # Binding rate constant (per concentration)
                    'k_off': float, # Unbinding rate constant
                    'cooperativity': float, # Optional cooperativity factor
                    'n_sites': int # Number of binding sites (default 1)
                }
            }
        logic_function (callable): Function (state_dict) -> bool/float.
            Determines the transcription rate or active status based on which TFs are bound.
    """
    
    def __init__(self, tf_configs, logic_function=None):
        """
        Initialize the multi-input promoter.
        
        Args:
            tf_configs (dict): Configuration of TFs (rates, sites).
            logic_function (callable, optional): Logic for activity. 
                Defaults to AND gate (active if all TFs are bound).
        """
        self.tf_configs = tf_configs
        self.tf_names = list(tf_configs.keys())
        self.n_tfs = len(self.tf_names)
        
        # Default logic: All TFs must be bound (AND logic)
        if logic_function is None:
            self.logic_function = self._default_and_logic
        else:
            self.logic_function = logic_function
            
        # State representation:
        # A dictionary mapping TF name to number of bound sites (integers)
        # For single sites, 0 or 1.
        
        # Current state: discrete state of the promoter
        self.current_state = {name: 0 for name in self.tf_names}
        
        # Pre-compute cooperativity if needed? 
        # For simplicity, we assume independent binding unless specified.
        # We handle transitions per TF.
        
    def _default_and_logic(self, state):
        """Default AND gate: Active only if ALL TFs have at least one site bound."""
        return all(state[name] > 0 for name in self.tf_names)

    def step(self, dt, concentrations):
        """
        Advance the promoter state stochastically.
        
        Args:
            dt (float): Time step.
            concentrations (dict): Current concentrations of TFs, e.g. {'TF_A': 10.0}.
            
        Returns:
            float: 1.0 if active, 0.0 if inactive (or fractional activity).
        """
        
        # We treat each TF's binding/unbinding as a separate independent process 
        # (mostly true for independent sites).
        # For competitive binding or complex cooperativity, we'd need a full rate matrix.
        # Here we approximate with separate updates for each TF channel.
        # This is valid for small dt.
        
        for name in self.tf_names:
            config = self.tf_configs[name]
            bound = self.current_state[name]
            n_sites = config.get('n_sites', 1)
            
            conc = concentrations.get(name, 0.0)
            k_on = config['k_on']
            k_off = config['k_off']
            
            # Binding
            if bound < n_sites:
                # Rate depends on free sites and concentration
                # r_bind = k_on * [TF] * (n - bound)
                r_bind = k_on * conc * (n_sites - bound)
                p_bind = 1 - np.exp(-r_bind * dt)
                
                if np.random.rand() < p_bind:
                    self.current_state[name] += 1
                    
            # Unbinding
            if bound > 0:
                # Rate depends on bound sites
                # r_unbind = k_off * bound
                r_unbind = k_off * bound
                p_unbind = 1 - np.exp(-r_unbind * dt)
                
                if np.random.rand() < p_unbind:
                    self.current_state[name] -= 1
                    
        # Check activity based on new state
        is_active = self.logic_function(self.current_state)
        
        return 1.0 if is_active else 0.0

    def get_state(self):
        """Return current binding state."""
        return self.current_state.copy()

    def reset(self):
        """Reset to empty state."""
        self.current_state = {name: 0 for name in self.tf_names}
