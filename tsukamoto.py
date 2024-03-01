import skfuzzy as skf


temperature = skf.var('temperature', 0, 100)  
control_action = skf.var('control_action', -1, 1)  


cold = skf.trimf(temperature, (0, 0, 25))
warm = skf.trimf(temperature, (20, 50, 80))
hot = skf.trimf(temperature, (75, 100, 100))


decrease_a_lot = skf.trimf(control_action, (-1, -1, -0.5))
decrease_a_little = skf.trimf(control_action, (-0.75, -0.25, 0))
maintain = skf.trimf(control_action, (-0.25, 0.25, 0.25))
increase_a_little = skf.trimf(control_action, (0, 0.75, 1))
increase_a_lot = skf.trimf(control_action, (0.5, 1, 1))


rule1 = skf.rule(cold, decrease_a_lot)
rule2 = skf.rule(skf.fuzzy_or(cold, warm), decrease_a_little)
rule3 = skf.rule(warm, maintain)
rule4 = skf.rule(skf.fuzzy_or(warm, hot), increase_a_little)
rule5 = skf.rule(hot, increase_a_lot)


control_system = skf.ControlSystem([rule1, rule2, rule3, rule4, rule5])


input_temperature = 65  


simulation = skf.ControlSystemSimulation(control_system)
simulation.input['temperature'] = input_temperature


simulation.compute()


output_action = skf.defuzzify(control_action, simulation.output['control_action'], 'centroid')

print(f"Input temperature: {input_temperature} Celsius")
print(f"Recommended control action: {output_action}")
