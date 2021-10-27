from .solver_train import TrainSolver
from .solver_test import TestSolver

def get_solver(args, config):

    solver = TrainSolver(args, config)

    #solver = TestSolver(args, config)

    
    return solver