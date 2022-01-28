import test_functions
from experiment import single_run, Config

def simple_1d():
    config_dict = {
        "test_function_name": test_functions.Simple1D.name,
        "n_initial_points": 3,
        "n_query_points": 4,
        "n_optimization_steps": 3,
        "n_repeats": 10
    }


    # config_dict["acquisition_method_name"] = "BatchMC"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)


    # config_dict["acquisition_method_name"] = "DistanceBased"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)

    config_dict["acquisition_method_name"] = "KB"
    config = Config.from_dict(config_dict)
    single_run(config, save_to_file=True)


def gardner():
    config_dict = {
        "test_function_name": test_functions.Gardner2D.name,
        "n_initial_points": 3,
        "n_query_points": 4,
        "n_optimization_steps": 3,
        "n_repeats": 10
    }


    # config_dict["acquisition_method_name"] = "BatchMC"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)


    # config_dict["acquisition_method_name"] = "DistanceBased"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)

    config_dict["acquisition_method_name"] = "KB"
    config = Config.from_dict(config_dict)
    single_run(config, save_to_file=True)


def hartmann_ackley():
    config_dict = {
        "test_function_name": test_functions.HartmannAckley6D.name,
        "n_initial_points": 3,
        "n_query_points": 4,
        "n_optimization_steps": 5,
        "n_repeats": 10
    }


    # config_dict["acquisition_method_name"] = "BatchMC"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)

    # config_dict["acquisition_method_name"] = "DistanceBased"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)

    config_dict["acquisition_method_name"] = "KB"
    config = Config.from_dict(config_dict)
    single_run(config, save_to_file=True)


def zdt3():
    config_dict = {
        "test_function_name": test_functions.ZDT3.name,
        "n_initial_points": 3,
        "n_query_points": 4,
        "n_optimization_steps": 5,
        "n_repeats": 10
    }

    # config_dict["acquisition_method_name"] = "BatchMC"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)

    # config_dict["acquisition_method_name"] = "DistanceBased"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)

    config_dict["acquisition_method_name"] = "KB"
    config = Config.from_dict(config_dict)
    single_run(config, save_to_file=True)



def DTLZ2_3_objectives():
    config_dict = {
        "test_function_name": test_functions.DTLZ2.name,
        "n_initial_points": 3,
        "n_query_points": 4,
        "n_optimization_steps": 5,
        "n_repeats": 10
    }


    # config_dict["acquisition_method_name"] = "BatchMC"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)

    # config_dict["acquisition_method_name"] = "DistanceBased"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)

    config_dict["acquisition_method_name"] = "KB"
    config = Config.from_dict(config_dict)
    single_run(config, save_to_file=True)


if __name__ == '__main__':
    simple_1d()
    gardner()
    hartmann_ackley()
    zdt3()
    DTLZ2_3_objectives()
