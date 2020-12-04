import numpy as np
import scipy.stats as stats


def generate(agent_average, agent_sigma, agent_num, control_average, control_sigma, control_num, start, end, iterations):
    values = []
    for x in range(iterations):
        agent_dist = stats.truncnorm((start - agent_average) / agent_sigma,
                                     (end - agent_average) / agent_sigma, loc=agent_average, scale=agent_sigma)
        agent_values = agent_dist.rvs(agent_num)
        control_dist = stats.truncnorm((start - control_average) / control_sigma,
                                       (end - control_average) / control_sigma, loc=control_average, scale=control_sigma)
        control_values = control_dist.rvs(control_num)
        u_statistic, p_val = stats.mannwhitneyu(agent_values, control_values)
        values.append(p_val)
    return round(np.mean(values), 20)


if __name__ == "__main__":
    structure = {
        "control": {
            "average": 2.30,
            "sigma": 1.69,
            "n": 35,
        },
        "structure_design": {
            "average": 5.07,
            "sigma": 2.38,
            "n": 35,
            "start": 0,
            "end": 12
        },
        "selective_attention": {
            "average": 3.31,
            "sigma": 2.25,
            "n": 35,
            "start": 0,
            "end": 12
        },
    }
    comprehension = {
        "control": {
            "average": 10.46,
            "sigma": 2.90,
            "n": 35,
        },
        "structure_design": {
            "average": 11.94,
            "sigma": 3.00,
            "n": 35,
            "start": 0,
            "end": 20
        },
        "selective_attention": {
            "average": 11.28,
            "sigma": 3.31,
            "n": 35,
            "start": 0,
            "end": 20
        },
    }
    result = {
        "structure": {
            "structure_design": generate(
                structure["structure_design"]["average"],
                structure["structure_design"]["sigma"],
                structure["structure_design"]["n"],
                structure["control"]["average"],
                structure["control"]["sigma"],
                structure["control"]["n"],
                structure["structure_design"]["start"],
                structure["structure_design"]["end"],
                100
            ),
            "selective_attention": generate(
                structure["selective_attention"]["average"],
                structure["selective_attention"]["sigma"],
                structure["selective_attention"]["n"],
                structure["control"]["average"],
                structure["control"]["sigma"],
                structure["control"]["n"],
                structure["selective_attention"]["start"],
                structure["selective_attention"]["end"],
                100
            )
        },
        "comprehension": {
            "structure_design": generate(
                comprehension["structure_design"]["average"],
                comprehension["structure_design"]["sigma"],
                comprehension["structure_design"]["n"],
                comprehension["control"]["average"],
                comprehension["control"]["sigma"],
                comprehension["control"]["n"],
                comprehension["structure_design"]["start"],
                comprehension["structure_design"]["end"],
                100
            ),
            "selective_attention": generate(
                comprehension["selective_attention"]["average"],
                comprehension["selective_attention"]["sigma"],
                comprehension["selective_attention"]["n"],
                comprehension["control"]["average"],
                comprehension["control"]["sigma"],
                comprehension["control"]["n"],
                comprehension["selective_attention"]["start"],
                comprehension["selective_attention"]["end"],
                100
            )
        }
    }
    print(result)
