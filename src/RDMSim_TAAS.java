import rdm.management.Effector;
import rdm.management.NetworkManagment;
import rdm.management.Probe;
import rdm.management.RDMSimulator;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class RDMSim_TAAS {
	private static final Logger log = LogManager.getLogger(RDMSim_TAAS.class);
	/* Class for configuring and running each component */
		private SolverProperties sp;     // object containing user-defined properties
	private Solver solver;           // the solver that we use to solve a POMDP, which is exact or approximate
	private ERPolicy erPolicy;       // the entropy-regularized policy that we use to select actions
	private String domainDirName;    // name of the directory containing .POMDP files
	private String domainDir;        // full path of the domain directory
	
	/** Optional experiment parameters (from solver.config): run seed, surprise measure, p_c, useSurpriseUpdating, lookback. Used for reproducible runs and paper experiments. */
	private int runSeed = 222;
	private String surpriseMeasureForGamma = "MIP";
	private double p_c = 0.5;
	private boolean useSurpriseUpdating = true;
	private int lookback = 5;
	/**
	 * Main entry point for running the RDMSim POMDP simulation
	 * @param args first argument should be a filename of a .POMDP file
	 */
    public static void main(String args[])
	{
		long startTime = System.currentTimeMillis();

		log.info("SolvePOMDP taken from the implementation in \"Information-Driven Adaptation and Entropy-Regularised Planning under Non-Stationarity\"");
		
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
