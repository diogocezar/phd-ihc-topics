import numpy as np
import scipy.stats as stats


def generate(agent_average, agent_sigma, agent_num, control_average, control_sigma, control_num, start, end, iterations):
    values = []
    for x in range(iterations):
        agent_dist = stats.truncnorm(
            (start - agent_average) / agent_sigma, (end - agent_average) / agent_sigma, loc=agent_average, scale=agent_sigma)
        agent_values = agent_dist.rvs(agent_num)
        control_dist = stats.truncnorm(
            (start - control_average) / control_sigma, (end - control_average) / control_sigma, loc=control_average, scale=control_sigma)
        control_values = control_dist.rvs(control_num)
        u_statistic, p_val = stats.mannwhitneyu(
            agent_values, control_values)
        values.append(p_val)
    return [values, np.mean(values)]


if __name__ == "__main__":
    values, result = generate(4.3, 1.3, 43, 4.6, 1.8, 46, 1, 7, 100)
    print(result)
