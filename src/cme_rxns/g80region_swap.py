"""
G80 region diffusion swap reactions: cytoplasm <=> nucleoplasm

symmetric rate constants

@param sim The simulation object to which the reactions will be added
"""
def getG80TransportReactions(sim):

    Kf80 = 50.0 # min^-1 (Ramsey 2006)
    Kr80 = 50.0 # min^-1

    # Monomer Transport
    # G80 -> G80C (Nucleus to Cytoplasm)
    sim.addReaction(reactant='G80', product='G80C', rate=Kf80)
    # G80C -> G80 (Cytoplasm to Nucleus)
    sim.addReaction(reactant='G80C', product='G80', rate=Kr80)

    # Dimer Transport
    # G80d -> G80Cd (Nucleus to Cytoplasm)
    sim.addReaction(reactant='G80d', product='G80Cd', rate=Kf80)
    # G80Cd -> G80d (Cytoplasm to Nucleus)
    sim.addReaction(reactant='G80Cd', product='G80d', rate=Kr80)
