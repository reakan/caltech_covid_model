### CoViD Model (Updated 200703)

Just to recap; each agent "owns" two Markov chains. One determines location and is completely independent and the other one determines state and is dependant on the agent's current location. The "location" chain is fully-connected and updated every 5 minutes with movement biased towards a "home" location. The state chain is updated once daily from a calculated "cumulative exposure." The state chain is shown below.

![State Transition Chain](chain_updated.png)

All model code has been put into a separate module for easier use.

Simulations are "run" on the Caltech campus with Chandler, Bechtel and each of the 3 North Houses (Lloyd, Page, Ruddock) being represented.

To demonstrate agent movement across campus, a smaller simulation was run for a shorter period of time (2 days, 30 people). Below is a visualization of student movement across the 5 locations outlined above for both lunch policies; lunch at a "home location" in a dorm vs in Chandler. 

![Chandler Shutdown](200701_movement.gif)

To determine outcomes, a full simulation with 300 people was run over a 14 day period. It was assumed 150 students are assigned to Bechtel house and 50 students are assigned to each of the North houses. Running the simulation with these many people is computationally expensive, so confidence intervals were drawn using only 5 trials (execution time ~ 6 hours).

![Test/No Test](200701_long_testing.gif)
(Sensitivity parameter had to be lowered substantially as testing is only done once per day)


Right now, I'm working on implementing different testing policies.