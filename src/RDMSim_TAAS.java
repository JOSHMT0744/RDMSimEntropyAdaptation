import rdm.management.Effector;
import rdm.management.NetworkManagment;
import rdm.management.Probe;
import rdm.management.RDMSimulator;

public class RDMSim_TAAS {
    public static void main(String args[])
	{
		
		//Step: 1 set the networkproperties and simulation properties by loading the simulation configuration
		NetworkManagment nm=new NetworkManagment();
		
		
		//Step: 2 Instantiate the Probe and effector
		Probe probe=nm.getProbe();
		Effector effector=nm.getEffector();
		
		//Step 3: Initialize the mape-K feedback loop using the Probe and effector
		MAPE_KLoop loop=new MAPE_KLoop(probe,effector);
		
		//Run simulation for the number of simulation runs defined to execute the feedback loop
		for(int timestep=0;timestep<nm.simulation_properties.getSimulationRuns();timestep++) {
			
			//start the feedback loop
			loop.run(timestep);
			
		
		
		}
		RDMSimulator.displayResults(args);
		
	}
}
