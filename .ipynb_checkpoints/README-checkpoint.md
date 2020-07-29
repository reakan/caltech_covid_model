### CoViD Model (Updated 200714)

Just to recap; each agent "owns" two Markov chains. One determines location and is completely independent and the other one determines state and is dependant on the agent's current location. The "location" chain is fully-connected and updated every 5 minutes with movement biased towards a "home" location. The state chain is updated once daily from a calculated "cumulative exposure." The state chain is shown below.

![State Transition Chain](chain_updated.png)

All model code has been put into a separate module for easier use.

Simulations are "run" on the Caltech campus with different locations being represented. The simulation testing lunch policy uses the following locations; Chandler, Bechtel and each of the 3 North Houses (Lloyd, Page, Ruddock). A standard simulation uses the following 10 locations; Broad, Moore, Watson, Beckman, Noyes, Braun, Schlinger, Kerckhoff, Spalding, Chandler.


To determine whether it's better for students to eat lunch in their dorms vs Chandler, a full simulation with 300 people was run over a 14 day period. It was assumed 150 students are assigned to Bechtel house and 50 students are assigned to each of the North houses. Running the simulation with these many people is computationally expensive, so confidence intervals were drawn using only 10 trials (execution time ~ 4 hours).

![Lunch Demo](chandler_simulation_FINAL5.png)

It's a little hard to see the difference between the two policies so I've plotted the differences between the two policies with bootstrapped confidence intervals below

![Lunch Policy Differences](chandler_policy_difference.png)


To explore different testing policies, I made the assumption that test results are delivered immediately (overnight) to simplify the math/code. One of the questions I wanted to answer (w.r.t a random testing policy) was how do the SEIRD populations change with respect to the number of people tested per day.

![Random Testing Policy](RL/random_policy_fig.png)

The second testing policy used state forecasting and selected individuals most likely to become exposed or infected the next day, given locations visited throughout the day and a partially-observable disease state. This policy was termed a "greedy" policy. Below is a comparison of the greedy and random testing policies. 

![Greedy Testing Policy](RL/greedy_policy.png)

Not much difference. I know the testing/quarantine code works because the plot two cells above had a horizontal line at the top. To test if the forecasting code worked, I adjusted the "sensitivity" (probability of contracting covid given exposure to an infected person) on the last 10 people in an environment. I then compared random and greedy testing policies with an allowed 10 tests per day. The number of times an individual was tested was recorded and the histograms are plotted below. Sensitivity among individuals is plotted as a red line.

![Greedy Testing History](RL/action_history.png)

The histogram on the random testing side seems uniform, which matches expectations. The histogram on the right shows clear preference for the individuals with higher sensitivity. These things put together leads me to believe that the state forecasting code is working properly, it's just that a naive implementation of a greedy algorithm doesn't work better than random testing for some reason. Maybe I need to just select the people most likely to become infected and not exposed to filter susceptible individuals.


Are there any specific figures to generate to send to the Gates Foundation people?



