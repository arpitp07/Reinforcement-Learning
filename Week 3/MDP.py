import mdptoolbox
import numpy as np

P = np.array([[[0.5,0.5],[0.4, 0.6]],[[0.8,0.2],[0.7,  0.3]]],dtype=float)
R = np.array([[[9.0,3.0],[3.0,-7.0]],[[4.0,4.0],[1.0,-19.0]]],dtype=float)
fh = mdptoolbox.mdp.FiniteHorizon(P, R, 1, 11)
fh.run()
fh.V

fh.policy