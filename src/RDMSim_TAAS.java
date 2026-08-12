import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Random;

import rdm.management.Effector;
import rdm.management.NetworkManagment;
import rdm.management.Probe;
import rdm.management.RDMSimulator;
import remotemirroring.RDMSimConnector;
import remotemirroring.RDMTransitionProb;
import remotemirroring.ResultsLog;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import solver.AlphaVector;
import solver.BeliefPoint;
import solver.ERPBVI;
import solver.Solver;
import solver.Perseus;
import solver.ERPerseus;
import solver.fastERPBVI;
import solver.ERPolicy;

import pomdp.POMDP;
import pomdp.PomdpParser;
import pomdp.SolverProperties;

public class RDMSim_TAAS {
	private static final Logger log = LogManager.getLogger(RDMSim_TAAS.class);
	/* Class for configuring and running each component */
	private SolverProperties sp; // object containing user-defined properties
	private Solver solver; // the solver that we use to solve a POMDP, which is exact or approximate
	private ERPolicy erPolicy; // the entropy-regularized policy that we use to select actions
	private String domainDirName; // name of the directory containing .POMDP files
	private String domainDir; // full path of the domain directory

	/**
	 * Optional experiment parameters (from solver.config): run seed, surprise
	 * measure, p_c, useSurpriseUpdating, lookback. Used for reproducible runs and
	 * paper experiments.
	 */
	private int runSeed = 222;
	private String surpriseMeasureForGamma = "MIP";
	private double p_c = 0.5;
	private boolean useSurpriseUpdating = true;
	private int lookback = 5;

	private double mon1Threshold = 20.0;
	private double mon2Threshold = 20.0;
	private double mon3Threshold = 20.0;

	public RDMSim_TAAS() {
		// read paramters from config file
		readConfigFile();

		// check if required directories exist
		configureDirectories();
	}

	/**
	 * Returns the configured output directory (from solver.config outputDirectory).
	 * Used when invoking createCharts.py.
	 */
	public String getOutputDir() {
		return sp.getOutputDir();
	}

	/**
	 * Returns mon1Threshold from solver.config (for createCharts.py MEC
	 * satisfaction plot).
	 */
	public double getMon1Threshold() {
		return mon1Threshold;
	}

	/**
	 * Returns mon2Threshold from solver.config (for createCharts.py RPL
	 * satisfaction plot).
	 */
	public double getMon2Threshold() {
		return mon2Threshold;
	}

	/**
	 * Returns mon3Threshold from solver.config (for createCharts.py RPL
	 * satisfaction plot).
	 */
	public double getMon3Threshold() {
		return mon3Threshold;
	}

	/**
	 * Helper method to get property with error handling
	 */
	private String getPropertyOrThrow(Properties properties, String key) {
		String value = properties.getProperty(key);
		if (value == null || value.trim().isEmpty()) {
			throw new RuntimeException("Missing or empty property '" + key + "' in solver.config");
		}
		return value.trim();
	}

	/**
	 * Optional property with default; used for experiment parameters (runSeed,
	 * surpriseMeasureForGamma, p_c).
	 */
	private String getProperty(Properties properties, String key, String defaultValue) {
		String value = properties.getProperty(key);
		if (value == null || value.trim().isEmpty()) {
			return defaultValue;
		}
		return value.trim();
	}

	/**
	 * Find the solver.config file path, handling both IDE and command-line
	 * execution.
	 * If -DconfigPath=<path> is set, that path is used (for experiment runners).
	 */
	private String findConfigFile() {
		String configPathOverride = System.getProperty("configPath");
		if (configPathOverride != null && !configPathOverride.isEmpty()) {
			File override = new File(configPathOverride);
			if (override.exists()) {
				return override.getAbsolutePath();
			}
		}
		// Try relative path first (works when running from project root)
		File configFile = new File("src/solver.config");
		if (configFile.exists()) {
			return configFile.getPath();
		}

		/*
		 * Try L4Project/src/solver.config (when running from workspace root)
		 * configFile = new File("/src/solver.config");
		 * if (configFile.exists()) {
		 * return configFile.getPath();
		 * }
		 */

		// Try using class location (works when running from JAR or compiled classes)
		try {
			String path = RDMSim_TAAS.class.getProtectionDomain().getCodeSource().getLocation().getPath();
			String decodedPath = URLDecoder.decode(path, "UTF-8");

			if (decodedPath.endsWith(".jar")) {
				// Running from JAR - config should be in same directory or src/
				int endIndex = decodedPath.lastIndexOf("/");
				String jarDir = decodedPath.substring(0, endIndex);
				configFile = new File(jarDir + "/src/solver.config");
				if (configFile.exists()) {
					return configFile.getPath();
				}
			} else {
				// Running from compiled classes - look for src/ relative to class location
				File classDir = new File(decodedPath);
				// Navigate up from bin/ to project root, then to src/
				File projectRoot = classDir.getParentFile().getParentFile();
				configFile = new File(projectRoot, "src/solver.config");
				if (configFile.exists()) {
					return configFile.getPath();
				}
			}
		} catch (Exception e) {
			// Fall through to default
		}

		// Default fallback
		return "src/solver.config";
	}

	/**
	 * Read the solver.config file. It creates a properties object and it
	 * initialises
	 */
	private void readConfigFile() {
		this.sp = new SolverProperties();

		Properties properties = new Properties();

		// Find config file relative to the class location (works from both IDE and
		// command line)
		String configPath = findConfigFile();

		try {
			FileInputStream file = new FileInputStream(configPath);
			properties.load(file);
			file.close();
		} catch (FileNotFoundException e) {
			log.error("Could not find solver.config at: {}; cwd={}", configPath, System.getProperty("user.dir"));
			throw new RuntimeException("solver.config file not found. Please ensure it exists in the src/ directory.",
					e);
		} catch (IOException e) {
			e.printStackTrace();
			throw new RuntimeException("Error reading solver.config file", e);
		}

		// Validate that properties were loaded
		if (properties.isEmpty()) {
			throw new RuntimeException("solver.config file is empty or could not be read");
		}

		// Exact Algorithm Settings
		sp.setEpsilon(Double.parseDouble(getPropertyOrThrow(properties, "epsilon")));

		// Directories (optional JVM override -DoutputDirectory=... for experiment
		// runners)
		String outputDirFromConfig = getPropertyOrThrow(properties, "outputDirectory");
		String outputDirOverride = System.getProperty("outputDirectory");
		sp.setOutputDirName(outputDirOverride != null && !outputDirOverride.trim().isEmpty() ? outputDirOverride.trim()
				: outputDirFromConfig);
		this.domainDirName = getPropertyOrThrow(properties, "domainDirectory");

		// Approximate Algorithm Settings
		sp.setBeliefSamplingRuns(Integer.parseInt(getPropertyOrThrow(properties, "beliefSamplingRuns")));
		sp.setBeliefSamplingSteps(Integer.parseInt(getPropertyOrThrow(properties, "beliefSamplingSteps")));

		// General Settings
		String algorithmType = getPropertyOrThrow(properties, "algorithmType");
		String lambda = getPropertyOrThrow(properties, "lambda");
		sp.setLambda(Double.parseDouble(lambda));
		sp.setTimeLimit(Double.parseDouble(getPropertyOrThrow(properties, "timeLimit")));
		sp.setValueFunctionTolerance(Double.parseDouble(getPropertyOrThrow(properties, "valueFunctionTolerance")));

		// Optional experiment parameters (for reproducible runs and paper experiments)
		this.runSeed = Integer.parseInt(getProperty(properties, "runSeed", "222"));
		this.surpriseMeasureForGamma = getProperty(properties, "surpriseMeasureForGamma", "MIP");
		this.p_c = Double.parseDouble(getProperty(properties, "p_c", "0.5"));
		if (this.surpriseMeasureForGamma != null && !this.surpriseMeasureForGamma.matches("CC|BF|MIP")) {
			throw new RuntimeException(
					"surpriseMeasureForGamma must be CC, BF, or MIP; got '" + this.surpriseMeasureForGamma + "'");
		}
		if (this.p_c <= 0 || this.p_c >= 1) {
			throw new RuntimeException("p_c must be in (0, 1); got " + this.p_c);
		}
		String useSurpriseUpdatingStr = getProperty(properties, "useSurpriseUpdating", "true");
		if (!useSurpriseUpdatingStr.equals("true") && !useSurpriseUpdatingStr.equals("false")) {
			throw new RuntimeException(
					"useSurpriseUpdating must be true or false; got '" + useSurpriseUpdatingStr + "'");
		}
		this.useSurpriseUpdating = useSurpriseUpdatingStr.equals("true");
		this.lookback = Integer.parseInt(getProperty(properties, "lookback", "5"));
		if (this.lookback <= 0) {
			throw new RuntimeException("lookback must be > 0; got " + this.lookback);
		}

		// Optional link failure injection
		/*
		 * String linkFailureTimestepStr = getProperty(properties,
		 * "linkFailureTimestep", "");
		 * if (linkFailureTimestepStr != null &&
		 * !linkFailureTimestepStr.trim().isEmpty()) {
		 * this.linkFailureTimestep = Integer.parseInt(linkFailureTimestepStr.trim());
		 * String linkFailureLinksStr = getProperty(properties, "linkFailureLinks", "");
		 * if (linkFailureLinksStr != null && !linkFailureLinksStr.trim().isEmpty()) {
		 * this.linkFailureLinksList = new ArrayList<>();
		 * for (String pair : linkFailureLinksStr.split(",")) {
		 * String[] parts = pair.trim().split("-");
		 * if (parts.length == 2) {
		 * this.linkFailureLinksList.add(new int[] {
		 * Integer.parseInt(parts[0].trim()),
		 * Integer.parseInt(parts[1].trim())
		 * });
		 * }
		 * }
		 * }
		 * String linkRecoveryTimestepStr = getProperty(properties,
		 * "linkRecoveryTimestep", "");
		 * if (linkRecoveryTimestepStr != null &&
		 * !linkRecoveryTimestepStr.trim().isEmpty()) {
		 * this.linkRecoveryTimestep = Integer.parseInt(linkRecoveryTimestepStr.trim());
		 * }
		 * }
		 */

		// Optional NFR thresholds for state discretisation (CHANGE DEFAULT VALUES HERE)
		String mon1ThresholdStr = getProperty(properties, "mon1Threshold", "20");
		String mon2ThresholdStr = getProperty(properties, "mon2Threshold", "20");
		String mon3ThresholdStr = getProperty(properties, "mon3Threshold", "20");
		this.mon1Threshold = Double.parseDouble(mon1ThresholdStr.trim());
		this.mon2Threshold = Double.parseDouble(mon2ThresholdStr.trim());
		this.mon3Threshold = Double.parseDouble(mon23ThresholdStr.trim());

		if (this.mon1Threshold < 0 || this.mon1Threshold > 100) {
			throw new RuntimeException("mecThreshold must be > 0; got " + this.mecThreshold);
		}
		if (this.mon2Threshold < 0 || this.mon2Threshold >= 100) {
			throw new RuntimeException("rplThreshold must be in (0, 1); got " + this.rplThreshold);
		}
		if (this.mon3Threshold < 0 || this.mon3Threshold >= 100) {
			throw new RuntimeException("mon3Threshold must be in (0, 1); got " + this.rplThreshold);
		}

		// Error checking solver.config parameters
		if (!algorithmType.equals("perseus") && !algorithmType.equals("gip") && !algorithmType.equals("erpbvi")
				&& !algorithmType.equals("erperseus")) {
			throw new RuntimeException("Unexpected algorithm type in properties file");
		}

		String dumpPolicyGraphStr = getPropertyOrThrow(properties, "dumpPolicyGraph");
		if (!dumpPolicyGraphStr.equals("true") && !dumpPolicyGraphStr.equals("false")) {
			throw new RuntimeException("Policy graph property must be either true or false");
		} else {
			sp.setDumpPolicyGraph(dumpPolicyGraphStr.equals("true") && algorithmType.equals("gip"));
		}

		String dumpActionLabelsStr = getPropertyOrThrow(properties, "dumpActionLabels");
		if (!dumpActionLabelsStr.equals("true") && !dumpActionLabelsStr.equals("false")) {
			throw new RuntimeException("Action label property must be either true or false");
		} else {
			sp.setDumpActionLabels(dumpActionLabelsStr.equals("true"));
		}

		log.info(
				"Solver parameters: epsilon={}, valueFunctionTolerance={}, timeLimit={}, beliefSamplingRuns={}, beliefSamplingSteps={}, lambda={}",
				sp.getEpsilon(), sp.getValueFunctionTolerance(), sp.getTimeLimit(), sp.getBeliefSamplingRuns(),
				sp.getBeliefSamplingSteps(), sp.getLambda());
		log.info(
				"Experiment: runSeed={}, surpriseMeasure={}, p_c={}, useSurpriseUpdating={}, lookback={}, mon1Threshold={}, mon2Threshold={}, mon3Threshold",
				runSeed, surpriseMeasureForGamma, p_c, useSurpriseUpdating, lookback, mon1Threshold, mon2Threshold,
				mon3Threshold);

		// load required POMDP algorithm (use runSeed for reproducible experiments)
		switch (algorithmType) {
			case "gip":
				throw new RuntimeException("GIP is not supported");
			case "perseus":
				this.solver = new Perseus(sp, new Random(runSeed));
				break;
			case "erperseus":
				this.solver = new ERPerseus(sp, new Random(runSeed), sp.getLambda());
				break;
			case "fasterpbvi":
				this.solver = new fastERPBVI(sp, new Random(runSeed), sp.getLambda(), false);
				break;
			case "erpbvi":
				// Entropy-Regularized PBVI with default parameters
				this.solver = new ERPBVI(sp, new Random(runSeed), sp.getLambda(), false);
				break;
			default:
				throw new RuntimeException("Unexpected algorithm type in properties file");
		}

		log.info("Algorithm: {}", algorithmType);
	}

	/**
	 * Find the domain directory by searching from current directory up to project
	 * root
	 */
	private File findDomainDirectory(File startDir, String domainDirName) {
		File current = startDir;
		int maxDepth = 5; // Prevent infinite loops
		int depth = 0;

		while (current != null && depth < maxDepth) {
			File domainDir = new File(current, domainDirName);
			if (domainDir.exists() && domainDir.isDirectory()) {
				return domainDir;
			}
			// Also check for L4Project/domains pattern
			File l4ProjectDir = new File(current, "L4Project");
			if (l4ProjectDir.exists() && l4ProjectDir.isDirectory()) {
				File domainDirInL4 = new File(l4ProjectDir, domainDirName);
				if (domainDirInL4.exists() && domainDirInL4.isDirectory()) {
					return domainDirInL4;
				}
			}
			current = current.getParentFile();
			depth++;
		}
		return null;
	}

	/**
	 * Checks if the desired domain and output directories exist, and it sets the
	 * full path to these directories.
	 */
	private void configureDirectories() {
		String path = RDMSim_TAAS.class.getProtectionDomain().getCodeSource().getLocation().getPath();
		String decodedPath = "";
		try {
			decodedPath = URLDecoder.decode(path, "UTF-8");
		} catch (UnsupportedEncodingException e) {
			log.error("Failed to decode path", e);
		}
		log.debug("Code source path: {}", decodedPath);

		if (decodedPath.endsWith(".jar")) {
			// solver has been started from jar, so we assume that output exists in the same
			// directory as the jar file
			int endIndex = decodedPath.lastIndexOf("/");
			String workingDir = decodedPath.substring(0, endIndex);
			sp.setWorkingDir(workingDir);
			domainDir = workingDir + "/" + domainDirName;
		} else {
			// solver has not been started from jar
			// Try to find the project root by looking for common project directories
			File currentDir = new File(System.getProperty("user.dir"));
			File domainDirFile = findDomainDirectory(currentDir, domainDirName);

			if (domainDirFile != null && domainDirFile.exists()) {
				domainDir = domainDirFile.getAbsolutePath();
				sp.setWorkingDir(domainDirFile.getParent());
			} else {
				// Fallback: assume current directory
				sp.setWorkingDir("");
				domainDir = domainDirName;
			}
		}

		File dir = new File(sp.getOutputDir());
		if (!dir.exists()) {
			boolean created = dir.mkdirs();
			if (!created) {
				throw new RuntimeException("Output directory could not be created");
			}
		} else if (!dir.isDirectory()) {
			throw new RuntimeException("Output path exists but is not a directory");
		}

		log.info("Output directory: {}; Domain directory: {}", sp.getOutputDir(), domainDir);
	}

	/**
	 * Method to run experiments for RDMSim case using POMDP
	 * 
	 * @param pomdpFileName
	 */
	public void runCaseRDM(String pomdpFileName) {
		try {
			// Use configured output directory instead of hardcoded path
			String outputDir = sp.getOutputDir();

			// Results Regression
			FileWriter fw_mc_regr = new FileWriter("MCRegressionResultsSolvePOMDP.txt");
			PrintWriter pw_mc_regr = new PrintWriter(fw_mc_regr);
			FileWriter fw_mr_regr = new FileWriter("MRRegressionResultsSolvePOMDP.txt");
			PrintWriter pw_mr_regr = new PrintWriter(fw_mr_regr);
			FileWriter fw_mp_regr = new FileWriter("MPRegressionResultsSolvePOMDP.txt");
			PrintWriter pw_mp_regr = new PrintWriter(fw_mp_regr);

			// read POMDP file
			File pomdpFile = new File(domainDir, pomdpFileName);
			if (!pomdpFile.exists()) {
				throw new RuntimeException("POMDP file not found: " + pomdpFile.getAbsolutePath() +
						"\nDomain directory: " + domainDir +
						"\nCurrent working directory: " + System.getProperty("user.dir"));
			}
			POMDP rdmPOMDP = PomdpParser.readPOMDP(pomdpFile.getAbsolutePath());

			// Here we should set the defined thresholds for the POMDP MON1-MON3
			//

			int mst_cnt = 0, rt_cnt = 0;
			RDMSimConnector rdmConnector = new RDMSimConnector();
			RDMTransitionProb rdmTransProbs = new RDMTransitionProb();

			// Get pre-defined scenario and its corresponding initial transitionFunction
			int currentscenario_case = RDMSimConnector.network_management.simulation_properties.getUncertaintyScenario()
					.getCurrentScenario();

			if (currentscenario_case == 0) {
				rdmPOMDP.transitionFunction = RDMTransitionProb.getTransitionFunction();
			} else {
				rdmPOMDP.transitionFunction = RDMTransitionProb.getTransitionFunctionCase(currentscenario_case);
			}

			RDMSimConnector.p = rdmPOMDP;

			// Beginning of MAPE-K loop
			for (RDMSimConnector.timestep = 0; RDMSimConnector.timestep < RDMSimConnector.network_management.simulation_properties
					.getSimulationRuns(); RDMSimConnector.timestep++) {

				// I really don't know what this `todeviate` stuff is about... The docs don't
				// explain it
				int todeviate = 2;
				System.out.println(todeviate);

				if (todeviate == 2) {
					// RDMTransitionProb.deviation_timesteps = (int)(Math.random() *
					// (RDMConfigurationConnected.deviation_timesteps_max-
					// RDMConfigurationConnected.deviation_timesteps_min + 1) +
					// RDMConfigurationConnected.deviation_timesteps_min);

					RDMTransitionProb.random_int = (int) (Math.random() * (12 - 9 + 1) + 9);
					// System.out.println("Random Number: "+RDMTransitionProb.random_int);
					// timestepcounter=0;
					///// For case 3
					RDMTransitionProb.random_int1 = (int) (Math.random() * (12 - 9 + 1) + 9);
					RDMTransitionProb.random_int2 = (int) (Math.random() * (12 - 9 + 1) + 9);

				} else {
					RDMTransitionProb.deviation_timesteps = 0;
					RDMTransitionProb.random_int = 0;
					// timestepcounter=0;

					RDMTransitionProb.random_int1 = 0;
					RDMTransitionProb.random_int2 = 0;
					System.out.println("no deviation for this timestep");
				}
				
				// update pomdp transition function according to the current scenario
				if(currentscenario_case==0) {
					rdmPOMDP.transitionFunction=RDMTransitionProb.getTransitionFunction();
				} else {
					rdmPOMDP.transitionFunction=RDMTransitionProb.getTransitionFunctionCase(currentscenario_case);
				}
				
				RDMSimConnector.p=rdmPOMDP;


				/*
				* MONITOR
				*/
				RDMSimConnector.monitorables=RDMSimConnector.network_management.getMonitorables();

				System.out.println("timestep: "+RDMSimConnector.timestep);
				

				/*
				* ANALYSE
				* Copmute belief state and satisfaction probabilites. 
				* The baseline state is implicitly available from the previous simulation run
				*/
				int cs = rdmPOMDP.getInitialStateRDM(); // CHECK HERE
				System.out.println("Initial state: "+cs);
				rdmPOMDP.setCurrentState(cs);
					
				System.out.println("current state: "+ rdmPOMDP.getCurrentState());

				// Get current belief over states (where each state represents the satisfaction combination for the NFRs)
				BeliefPoint initialbelief=rdmPOMDP.getInitialBelief();
				double b[]=initialbelief.getBelief();
				System.out.println("Initial Belief: "+b[0]+" "+b[1]+" "+b[2]+" "+b[3]+" "+b[4]+" "+b[5]+" "+b[6]+" "+b[7]);
				double mcsatprob=b[0]+b[1]+b[2]+b[3];
				double mrsatprob=b[0]+b[1]+b[4]+b[5];
				double mpsatprob=b[0]+b[2]+b[4]+b[6];
				
				////Results Log Regression////////
				
				pw_mc_regr.println(ResultsLog.bandwidthconsumption+","+mcsatprob+","+ResultsLog.satmc);
				pw_mr_regr.println(ResultsLog.activelinks+","+mrsatprob+","+ResultsLog.satmr);


				/*
				* PLAN
				* Solve the POMDP to determine the optimal adaptation action given the current belief state
				* Each AlphaVector encodes a lienar function over beliefs
				*/
				ArrayList<AlphaVector> V1=solver.solve(rdmPOMDP);
			 	System.out.println("Value size: "+V1.size()+"  Action label: "+ V1.get(0).getAction());

				// select action using the relevant selected action-selection policy
				int selectedAction;
				if (solver instanceof ERPBVI) {
					// ERPBVI has Q-functions directly available
					erPolicy = new ERPolicy(rdmPOMDP, (ERPBVI)solver, new Random(runSeed));
					selectedAction = erPolicy.selectAction(rdmPOMDP.getInitialBelief());
				} else if (solver instanceof ERPerseus) {
					// ERPerseus: extract Q-functions from value function
					double lambda = ((ERPerseus) solver).getLambda();
					erPolicy = new ERPolicy(rdmPOMDP, V1, lambda, new Random(runSeed));
					selectedAction = erPolicy.selectAction(rdmPOMDP.getInitialBelief());
				} else {
					int bestIndex = AlphaVector.getBestVectorIndex(rdmPOMDP.getInitialBelief().getBelief(), V1);
					selectedAction = V1.get(bestIndex).getAction();
				}
				log.debug("Selected action: {}", selectedAction);

				/*
				* EXECUTE
				* Execute the selected action (changing topology)
				* This also updates POMDP beliefs and transition probabilities
				*/

				// Check below is correct with index matching actions
				if(selectedAction==0) {
					mst_cnt++;
				} else {
					rt_cnt++;
				}

				rdmPOMDP.setInitialBelief(initialbelief);
				RDMSimConnector.p = rdmPOMDP;

				// Check Perform Action
				rdmConnector.performAction(selectedAction);
				rdmPOMDP = RDMSimConnector.p;
				System.out.println("Current State: "+pomdp.getCurrentState());
			
			System.out.println("\nTopology Count:: MST: "+mst_cnt+" RT: "+rt_cnt);
			}

			pw_mc_regr.flush();
			pw_mp_regr.flush();
			pw_mr_regr.flush();
			pw_mc_regr.close();
			pw_mr_regr.close();
			pw_mp_regr.close();
		} catch (IOException ioex) {
			log.error("IOException in runCaseIoT", ioex);
		} catch (RuntimeException ex) {
			log.error("Unexpected exception in runCaseIoT", ex);
		}
	}

	/**
	 * Solve a POMDP defined by a .POMDP file
	 * 
	 * @param pomdpFileName filename of a domain in the domain directory
	 */
	public void run(String pomdpFileName) {
		if (pomdpFileName.equals("IoT.POMDP")) {
			// This is where the original DeltaIoT implementation would have gone
		}
		if (pomdpFileName.equals("RDM.POMDP")) {
			runCaseRDM(pomdpFileName); // RDMSim case
		}
	}

	/**
	 * Main entry point for running the RDMSim POMDP simulation
	 * 
	 * @param args first argument should be a filename of a .POMDP file
	 */
	public static void main(String args[]) {
		long startTime = System.currentTimeMillis();

		log.info(
				"SolvePOMDP taken from the implementation in \"Information-Driven Adaptation and Entropy-Regularised Planning under Non-Stationarity\"");
		if (args.length == 0) {
			log.info("First argument should be the name of a file in the domains directory");
		}

		// Step: 1 set the networkproperties and simulation properties by loading the
		// simulation configuration
		// NetworkManagment nm = new NetworkManagment();

		// Step: 2 Instantiate the Probe and effector
		// Probe probe = nm.getProbe();
		// Effector effector = nm.getEffector();

		// Step 3: Initialize the mape-K feedback loop using the Probe and effector
		// MAPE_KLoop loop = new MAPE_KLoop_init(probe, effector);

		// Initialise POMDP
		RDMSim_TAAS pomdp = new RDMSim_TAAS();
		pomdp.run("rdm.POMDP");

		// Run simulation for the number of simulation runs defined to execute the
		// feedback loop
		/*
		 * for (int timestep = 0; timestep <
		 * nm.simulation_properties.getSimulationRuns(); timestep++) {
		 * 
		 * // start the feedback loop
		 * loop.run(timestep);
		 * 
		 * }
		 * RDMSimulator.displayResults(args);
		 */

		long endTime = System.currentTimeMillis();
		long totalTime = endTime - startTime;
		double totalTimeSeconds = totalTime / 1000.0;
		log.info("Total execution time: {} seconds", String.format("%.2f", totalTimeSeconds));

	}
}
