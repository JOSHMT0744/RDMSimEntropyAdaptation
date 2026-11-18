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
import rdm.network.Monitorables;
import rdm.simulation.UncertaintyScenario;
import remotemirroring.CRNStream;
import remotemirroring.OutputLogger;
import remotemirroring.RDMSimConnector;
import remotemirroring.RDMTransitionProb;
import remotemirroring.ScenarioOverride;
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
	private String algorithmType = "erperseus";
	private String surpriseMeasureForGamma = "MIP";
	private double p_c = 0.5;
	private boolean useSurpriseUpdating = true;
	private int lookback = 4;   // keep in sync with solver.config and RDMSimConnector

	/**
	 * NFR satisfaction thresholds, in absolute units. These are NOT read from
	 * solver.config: they are derived by RDMSim from the percentages in
	 * config_log_files/configuration.json and read back in runCaseRDM().
	 * Left at 0 until then so a failed read-back is detectable.
	 */
	private double mon1Threshold = 0.0;
	private double mon2Threshold = 0.0;
	private double mon3Threshold = 0.0;

	public RDMSim_TAAS() {
		// read paramters from config file
		readConfigFile();

		// check if required directories exist
		configureDirectories();
	}

	/**
	 * Find Python executable in virtual environment
	 */
	private static String findPythonExecutable() {
		// Try Windows path first
		File venvWindows = new File(".venv/Scripts/python.exe");
		if (venvWindows.exists()) {
			return venvWindows.getPath();
		}
		
		// Try Linux/Mac path
		File venvUnix = new File(".venv/bin/python");
		if (venvUnix.exists()) {
			return venvUnix.getPath();
		}
		
		// Try from L4Project directory
		File venvL4Windows = new File("./.venv/Scripts/python.exe");
		if (venvL4Windows.exists()) {
			return venvL4Windows.getPath();
		}
		
		File venvL4Unix = new File("./.venv/bin/python");
		if (venvL4Unix.exists()) {
			return venvL4Unix.getPath();
		}
		
		return null;
	}

	/**
	 * Find createCharts.py script
	 */
	private static String findChartsScript() {
		File script = new File("createCharts.py");
		if (script.exists()) {
			return script.getPath();
		}
		
		File scriptL4 = new File("./createCharts.py");
		if (scriptL4.exists()) {
			return scriptL4.getPath();
		}
		
		return null;
	}

	/**
	 * Run createCharts.py to generate graphs from the solver output.
	 * @param outputDirForCharts Directory where the solver wrote output (MECSattimestep.txt, gamma.txt, etc.);
	 *                           should match solver.config outputDirectory. Passed to createCharts.py as --output-dir.
	 * @param mecThreshold MEC threshold from solver.config (--mec-threshold).
	 * @param rplThreshold RPL threshold from solver.config (--rpl-threshold).
	 */
	public static void runPython(String outputDirForCharts, double mon1Threshold, double mon2Threshold, double mon3Threshold, int lookback) throws Exception {
		// Try to find Python executable in virtual environment
		String pythonPath = findPythonExecutable();
		if (pythonPath == null) {
			log.warn("Python virtual environment not found. Skipping chart generation. Expected: .venv\\Scripts\\python.exe or .venv/bin/python");
			return;
		}
		
		// Find createCharts.py relative to project root
		String chartsScript = findChartsScript();
		if (chartsScript == null) {
			log.warn("createCharts.py not found. Skipping chart generation.");
			return;
		}
		
		// Resolve output dir to absolute path so createCharts.py reads from the correct run directory (e.g. init_runs/s222)
		String outputDirAbs = new File(outputDirForCharts).getAbsolutePath();
		log.info("Running createCharts.py with --output-dir {} --mon1-threshold {} --mon2-threshold {} --mon3-threshold {} --lookback {}", outputDirAbs, mon1Threshold, mon2Threshold, mon3Threshold, lookback);
		ProcessBuilder pb = new ProcessBuilder(
			pythonPath, chartsScript,
			"--output-dir", outputDirAbs,
			"--mon1-threshold", String.valueOf(mon1Threshold),
			"--mon2-threshold", String.valueOf(mon2Threshold),
			"--mon3-threshold", String.valueOf(mon3Threshold),
			"--lookback", String.valueOf(lookback)
		);
		pb.redirectErrorStream(true);
		Process p = pb.start();
		
		BufferedReader reader = new BufferedReader(
				new InputStreamReader(p.getInputStream())
				);
		
		String line;
		while ((line = reader.readLine()) != null) {
			log.debug("PYTHON: {}", line);
		}
		p.waitFor();
	}

	/**
	 * Returns the configured output directory (from solver.config outputDirectory).
	 * Used when invoking createCharts.py.
	 */
	public String getOutputDir() {
		return sp.getOutputDir();
	}

	/**
	 * Returns the MON1 (active links) threshold in absolute units, as derived by
	 * RDMSim from link_threshold in config_log_files/configuration.json. Only
	 * populated once runCaseRDM() has read it back from the simulator.
	 */
	public double getMon1Threshold() {
		return mon1Threshold;
	}

	/**
	 * Returns the MON2 (bandwidth consumption, Gbps) threshold in absolute units,
	 * as derived by RDMSim from bandwidth_threshold in
	 * config_log_files/configuration.json.
	 */
	public double getMon2Threshold() {
		return mon2Threshold;
	}

	/**
	 * Returns the MON3 (time to write, ms) threshold in absolute units, as derived
	 * by RDMSim from writing_times_threshold in
	 * config_log_files/configuration.json.
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
		 * Try RDMSIM_TAAS/src/solver.config (when running from workspace root)
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
		this.algorithmType = algorithmType;
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
		this.lookback = Integer.parseInt(getProperty(properties, "lookback", "4"));
		if (this.lookback <= 0) {
			throw new RuntimeException("lookback must be > 0; got " + this.lookback);
		}

		// Strength of the informative Dirichlet prior seeded from the .POMDP transition
		// function. 1.0 keeps the same total pseudo-count mass as a flat alpha=1 prior.
		PomdpParser.priorStrength = Double.parseDouble(getProperty(properties, "priorStrength", "1.0"));
		if (PomdpParser.priorStrength <= 0) {
			throw new RuntimeException("priorStrength must be > 0; got " + PomdpParser.priorStrength);
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

		// NFR thresholds are not read here: config_log_files/configuration.json is the
		// single source of truth, and runCaseRDM() reads the absolute values back out of
		// RDMSim once the simulator has been constructed.

		// Error checking solver.config parameters
		if (!algorithmType.equals("perseus") && !algorithmType.equals("gip") && !algorithmType.equals("erpbvi")
				&& !algorithmType.equals("erperseus") && !algorithmType.equals("fasterpbvi")) {
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
		log.info("Dirichlet priorStrength={}", PomdpParser.priorStrength);
		log.info(
				"Experiment: runSeed={}, surpriseMeasure={}, p_c={}, useSurpriseUpdating={}, lookback={}",
				runSeed, surpriseMeasureForGamma, p_c, useSurpriseUpdating, lookback);

		// load required POMDP algorithm (use runSeed for reproducible experiments)
		switch (algorithmType) {
			case "gip" -> throw new RuntimeException("GIP is not supported");
			case "perseus" -> this.solver = new Perseus(sp, new Random(runSeed));
			case "erperseus" -> this.solver = new ERPerseus(sp, new Random(runSeed), sp.getLambda());
			case "fasterpbvi" -> this.solver = new fastERPBVI(sp, new Random(runSeed), sp.getLambda(), false);
			case "erpbvi" -> // Entropy-Regularized PBVI with default parameters
				this.solver = new ERPBVI(sp, new Random(runSeed), sp.getLambda(), false);
			default -> throw new RuntimeException("Unexpected algorithm type in properties file");
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

			File rootDir = new File(current, "/"); // Check here. Potentially instead of "/" put "RDMSIM_TAAS"
			if (rootDir.exists() && rootDir.isDirectory()) {
				File rootFile = new File(rootDir, domainDirName);
				if (rootFile.exists() && rootFile.isDirectory()) {
					return rootFile;
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
			System.out.println("Running RDM case...");
			// Use configured output directory instead of hardcoded path
			String outputDir = sp.getOutputDir();

			// Results Regression
			FileWriter fw_mc_regr = new FileWriter(new File(outputDir,"MCRegressionResultsSolvePOMDP.txt"));
			PrintWriter pw_mc_regr = new PrintWriter(fw_mc_regr);
			FileWriter fw_mr_regr = new FileWriter(new File(outputDir,"MRRegressionResultsSolvePOMDP.txt"));
			PrintWriter pw_mr_regr = new PrintWriter(fw_mr_regr);
			FileWriter fw_mp_regr = new FileWriter(new File(outputDir,"MPRegressionResultsSolvePOMDP.txt"));
			PrintWriter pw_mp_regr = new PrintWriter(fw_mp_regr);

			// read POMDP file
			File pomdpFile = new File(domainDir, pomdpFileName);
			if (!pomdpFile.exists()) {
				throw new RuntimeException("POMDP file not found: " + pomdpFile.getAbsolutePath() +
						"\nDomain directory: " + domainDir +
						"\nCurrent working directory: " + System.getProperty("user.dir"));
			}
			POMDP rdmPOMDP = PomdpParser.readPOMDP(pomdpFile.getAbsolutePath());

			int mst_cnt = 0, rt_cnt = 0;

			// End-of-run summary tallies (per-NFR reporting).
			int mon1SatCount = 0, mon2SatCount = 0, mon3SatCount = 0;
			double mon1ProbSum = 0.0, mon2ProbSum = 0.0, mon3ProbSum = 0.0;
			double activeLinksSum = 0.0, bandwidthSum = 0.0, timeToWriteSum = 0.0;
			int observedSteps = 0;
			int stateChanges = 0;
			RDMSimConnector rdmConnector = new RDMSimConnector();
			RDMTransitionProb rdmTransProbs = new RDMTransitionProb();

			// Apply -Dscenario / -Dtimesteps now that RDMSim has parsed its own config.
			// The ablation driver runs many cells concurrently from one working directory, so
			// it cannot edit the shared config_log_files/configuration.json per cell; these
			// JVM properties are how a cell selects its scenario and horizon instead.
			int scenarioInForce = ScenarioOverride.apply(RDMSimConnector.network_management.simulation_properties);

			// RDMSim picked the t=0 topology at load time from the JSON's scenario, with an
			// unseedable coin flip for scenarios 3/6. Re-pick it from the scenario actually in
			// force, seeded by runSeed, whenever a cell must be reproducible: under -Dscenario
			// (the JSON's rule would be the wrong scenario's) or under CRN (an unseeded start
			// would break the pairing). A plain run keeps the exemplar's own behaviour.
			String scenarioProperty = System.getProperty("scenario");
			boolean crnRequested = "true".equalsIgnoreCase(System.getProperty("crn", "false"));
			if ((scenarioProperty != null && !scenarioProperty.trim().isEmpty()) || crnRequested) {
				ScenarioOverride.applyInitialTopology(scenarioInForce, runSeed);
			}

			// config_log_files/configuration.json is the single source of truth for the NFR
			// thresholds. NetworkProperties.loadNetworkSettings() (called from the
			// NetworkManagment constructor) converts the configured percentages into absolute
			// values against the network capacity implied by mirror_number:
			//   number_of_links       = 25*24/2 = 300
			//   link_threshold        35% of 300      = 105 active links
			//   bandwidth_threshold   40% of 300*30   = 3600 Gbps
			//   writing_times_threshold 45% of 300*20 = 2700 ms
			// POMDP.getInitialStateRDM() compares the measured monitorables against these
			// (al >= MON1, bw <= MON2, ttw <= MON3), so we only read them back here for
			// logging and chart generation - never overwrite them.
			Monitorables m = RDMSimConnector.network_management.network_properties.m;
			this.mon1Threshold = m.getThresholdActiveLinks();
			this.mon2Threshold = m.getThresholdBandwidthConsumption();
			this.mon3Threshold = m.getThresholdTimeToWrite();

			if (mon1Threshold <= 0 || mon2Threshold <= 0 || mon3Threshold <= 0) {
				throw new RuntimeException("NFR thresholds not loaded from config_log_files/configuration.json"
						+ " (got " + mon1Threshold + "/" + mon2Threshold + "/" + mon3Threshold + ")."
						+ " Run with the working directory set to the RDMSim_TAAS project root; cwd="
						+ System.getProperty("user.dir"));
			}
			log.info(
					"NFR thresholds from configuration.json: MON1(active links)>={}, MON2(bandwidth Gbps)<={}, MON3(time to write ms)<={}",
					mon1Threshold, mon2Threshold, mon3Threshold);

			// Wire the experiment parameters read from solver.config into the connector
			rdmConnector.setOutputDirectory(outputDir);
			rdmConnector.setSurpriseMeasureForGamma(surpriseMeasureForGamma);
			rdmConnector.setP_c(p_c);
			rdmConnector.setLookback(lookback);
			rdmConnector.setUseSurpriseUpdating(useSurpriseUpdating);

			// Set up monitorable logging: truncate any stale files from a previous run
			OutputLogger outputLogger = new OutputLogger(outputDir);
			String[] logFiles = { "monitorables.txt", "MON1Sat.txt", "MON1SatProb.txt", "MON2Sat.txt",
					"MON2SatProb.txt", "MON3Sat.txt", "MON3SatProb.txt", "SelectedAction.txt",
					"state_transitions.txt", "surpriseBF.txt", "surpriseCC.txt", "surpriseMIP.txt",
					"gamma.txt", "MIPBounds.txt", "runSummary.txt", "qvalues.txt" };
			for (String f : logFiles) {
				outputLogger.clearFile(f);
			}
			// The solver appends its value function every timestep; start from empty.
			outputLogger.clearFile(new File(pomdpFileName).getName().replace(".POMDP", "") + ".alpha");
			outputLogger.writeMonitorablesHeader();

			// Get pre-defined scenario and its corresponding initial transitionFunction
			int currentscenario_case = RDMSimConnector.network_management.simulation_properties.getUncertaintyScenario()
					.getCurrentScenario();

			log.info("Uncertainty scenario: {}", currentscenario_case);

			// Deviation magnitudes for the active scenario, read from configuration.json
			// rather than the hardcoded 9..12 that used to sit in the loop.
			UncertaintyScenario us = RDMSimConnector.network_management.simulation_properties.getUncertaintyScenario();
			int deviationMin = us.getMinDeviationMSTLinks();
			int deviationMax = us.getMaxDeviationMSTLinks();
			if (deviationMax < deviationMin) {
				deviationMax = deviationMin;
			}
			log.info("Deviation range for scenario {}: [{}, {}]", currentscenario_case, deviationMin, deviationMax);

			// The planning model is NOT set from RDMTransitionProb. It is the normalised
			// Dirichlet posterior (transitionBeliefCurr), seeded from the .POMDP prior and
			// updated each timestep by the SMiLe/surprise machinery in RDMSimConnector.
			// Overwriting transitionFunction here would hand the planner a hand-coded model
			// and make the whole learning path inert.
			rdmPOMDP.refreshLearnedModel();

			RDMSimConnector.p = rdmPOMDP;

			// Single policy RNG for the whole run. Constructing this inside the loop
			// reset the stream to runSeed every timestep, which collapses the stochastic
			// policy to a deterministic threshold rule once lambda > 0.
			Random policyRandom = new Random(runSeed);

			// Deviation RNG, seeded so runs are reproducible (Math.random() was not).
			Random deviationRandom = new Random(runSeed);

			// Common random numbers. RDMSim draws its monitorables with ThreadLocalRandom,
			// which cannot be seeded from outside the jar, so two ablation arms otherwise
			// face different environments and any difference between them is confounded with
			// noise. With -Dcrn=true the environment is pre-drawn for every timestep and BOTH
			// topologies from a (scenario, seed) stream, and every arm in that cell sees the
			// same numbers. Off by default so a plain run keeps the exemplar's own behaviour.
			int horizon = RDMSimConnector.network_management.simulation_properties.getSimulationRuns();
			CRNStream crnStream = null;
			if ("true".equalsIgnoreCase(System.getProperty("crn", "false"))) {
				crnStream = new CRNStream(currentscenario_case, runSeed, horizon,
						RDMSimConnector.network_management.simulation_properties.getUncertaintyScenario());
				log.info("Common random numbers ENABLED (scenario={}, seed={}, horizon={})",
						currentscenario_case, runSeed, horizon);
			} else {
				log.info("Common random numbers disabled; monitorables come from RDMSim's ThreadLocalRandom");
			}

			// The state/action of the previous iteration. RDMSim only reveals the effect of
			// an action on the NEXT getMonitorables(), so the transition (prevState,
			// prevAction) -> cs can only be recorded once cs has been monitored.
			int prevState = -1;
			int prevAction = -1;

			// Beginning of MAPE-K loop
			for (RDMSimConnector.timestep = 0; RDMSimConnector.timestep < RDMSimConnector.network_management.simulation_properties
					.getSimulationRuns(); RDMSimConnector.timestep++) {

				// Per-timestep deviation magnitudes used by RDMTransitionProb's scenario
				// cases. Seeded from runSeed so a run is reproducible; Math.random() here
				// previously made runs irreproducible despite runSeed being plumbed through
				// everywhere else.
				RDMTransitionProb.random_int = deviationMin + deviationRandom.nextInt(deviationMax - deviationMin + 1);
				RDMTransitionProb.random_int1 = deviationMin + deviationRandom.nextInt(deviationMax - deviationMin + 1);
				RDMTransitionProb.random_int2 = deviationMin + deviationRandom.nextInt(deviationMax - deviationMin + 1);



				/*
				* MONITOR
				*/
				RDMSimConnector.monitorables=RDMSimConnector.network_management.getMonitorables();

				// Replace the jar's unseedable draw with this cell's pre-drawn environment, so
				// that every ablation arm sharing (scenario, seed) monitors identical values
				// wherever its action path agrees. Must happen before ANALYSE: both the probe's
				// own fields and the Monitorables object are overwritten, because
				// POMDP.getInitialStateRDM() reads the former and the logging below reads the
				// latter.
				if (crnStream != null) {
					crnStream.overwrite(RDMSimConnector.timestep,
							RDMSimConnector.probe.getCurrentTopology().getTopologyName(),
							RDMSimConnector.monitorables);
				}

				outputLogger.appendRow("monitorables.txt", RDMSimConnector.timestep,
						RDMSimConnector.monitorables.getActiveLinks(),
						RDMSimConnector.monitorables.getBandwidthConsumption(),
						RDMSimConnector.monitorables.getTimeToWrite(),
						RDMSimConnector.monitorables.getAlpha(),
						RDMSimConnector.monitorables.getThresholdActiveLinks(),
						RDMSimConnector.monitorables.getThresholdBandwidthConsumption(),
						RDMSimConnector.monitorables.getThresholdTimeToWrite());

				System.out.println("timestep: "+RDMSimConnector.timestep);
				

				/*
				* ANALYSE
				* Copmute belief state and satisfaction probabilites. 
				* The baseline state is implicitly available from the previous simulation run
				*/
				int cs = rdmPOMDP.getInitialStateRDM();
				System.out.println("Observed state: "+cs);

				/*
				 * LEARN
				 * The monitorables just read are the consequence of the action executed on
				 * the previous timestep, so this is the point at which the transition
				 * (prevState, prevAction) -> cs is known. Fold it into the Dirichlet counts
				 * (surprise + SMiLe), refresh the planning model, and advance the belief
				 * filter. Doing this at EXECUTE time instead reads a stale probe and records
				 * a self-transition every step.
				 */
				if (RDMSimConnector.timestep > 0) {
					rdmConnector.observeTransition(prevAction, cs);
					rdmPOMDP = RDMSimConnector.p;

					double[] pb = rdmPOMDP.getInitialBelief().getBelief();
					outputLogger.appendRow("state_transitions.txt", RDMSimConnector.timestep - 1, prevState,
						prevAction, cs, pb[0], pb[1], pb[2], pb[3], pb[4], pb[5], pb[6], pb[7]);
					if (prevState != cs) {
						stateChanges++;
					}
				}

				rdmPOMDP.setCurrentState(cs);

				// Get current belief over states (each state is an NFR satisfaction combination)
				BeliefPoint initialbelief=rdmPOMDP.getInitialBelief();
				double b[]=initialbelief.getBelief();
				System.out.println("Belief: "+b[0]+" "+b[1]+" "+b[2]+" "+b[3]+" "+b[4]+" "+b[5]+" "+b[6]+" "+b[7]);
				// State encoding is (MC, MR, MP) = (bandwidth ok, links ok, ttw ok).
				// Note the naming: mc -> MON2 (bandwidth), mr -> MON1 (links), mp -> MON3 (ttw).
				double mcsatprob=b[0]+b[1]+b[2]+b[3];
				double mrsatprob=b[0]+b[1]+b[4]+b[5];
				double mpsatprob=b[0]+b[2]+b[4]+b[6];

				////Results Log Regression////////

				pw_mc_regr.println(ResultsLog.bandwidthconsumption+","+mcsatprob+","+ResultsLog.satmc);
				pw_mr_regr.println(ResultsLog.activelinks+","+mrsatprob+","+ResultsLog.satmr);
				pw_mp_regr.println(ResultsLog.timetowrite+","+mpsatprob+","+ResultsLog.satmp);

				// Running per-NFR tallies for the end-of-run summary.
				mon1SatCount += ResultsLog.satmr;
				mon2SatCount += ResultsLog.satmc;
				mon3SatCount += ResultsLog.satmp;
				mon1ProbSum += mrsatprob;
				mon2ProbSum += mcsatprob;
				mon3ProbSum += mpsatprob;
				activeLinksSum += ResultsLog.activelinks;
				bandwidthSum += ResultsLog.bandwidthconsumption;
				timeToWriteSum += ResultsLog.timetowrite;
				observedSteps++;

				// MON1 = active links, MON2 = bandwidth consumption, MON3 = time to write.
				// Each MONnSat row is "timestep measuredValue satisfiedFlag".
				outputLogger.appendRow("MON1Sat.txt", RDMSimConnector.timestep, ResultsLog.activelinks, ResultsLog.satmr);
				outputLogger.appendRow("MON1SatProb.txt", RDMSimConnector.timestep, mrsatprob);
				outputLogger.appendRow("MON2Sat.txt", RDMSimConnector.timestep, ResultsLog.bandwidthconsumption, ResultsLog.satmc);
				outputLogger.appendRow("MON2SatProb.txt", RDMSimConnector.timestep, mcsatprob);
				outputLogger.appendRow("MON3Sat.txt", RDMSimConnector.timestep, ResultsLog.timetowrite, ResultsLog.satmp);
				outputLogger.appendRow("MON3SatProb.txt", RDMSimConnector.timestep, mpsatprob);


				/*
				* PLAN
				* Solve the POMDP to determine the optimal adaptation action for the belief
				*/
				ArrayList<AlphaVector> V1=solver.solve(rdmPOMDP);
				System.out.println("Value size: "+V1.size());

				// select action using the relevant selected action-selection policy
				int selectedAction;
				if (solver instanceof ERPBVI) {
					erPolicy = new ERPolicy(rdmPOMDP, (ERPBVI)solver, policyRandom);
					selectedAction = erPolicy.selectAction(initialbelief);
				} else if (solver instanceof ERPerseus) {
					double lambda = ((ERPerseus) solver).getLambda();
					erPolicy = new ERPolicy(rdmPOMDP, V1, lambda, policyRandom);
					selectedAction = erPolicy.selectAction(initialbelief);
				} else {
					int bestIndex = AlphaVector.getBestVectorIndex(initialbelief.getBelief(), V1);
					selectedAction = V1.get(bestIndex).getAction();
				}
				// Log per-action Q-values: their spread is the scale lambda must be set against.
				if (erPolicy != null) {
					double[] qv = erPolicy.getQValues(initialbelief.getBelief());
					if (qv.length >= 2) {
						outputLogger.appendRow("qvalues.txt", RDMSimConnector.timestep, qv[0], qv[1], Math.abs(qv[0] - qv[1]));
					}
				}
				log.debug("Selected action: {}", selectedAction);
				outputLogger.appendRow("SelectedAction.txt", RDMSimConnector.timestep, selectedAction);

				/*
				* EXECUTE
				* Apply the action (changing topology). Its effect is only visible at the next
				* getMonitorables(), and is folded into the model by the LEARN block above on
				* the following iteration.
				*/
				if(selectedAction==0) {
					mst_cnt++;
				} else {
					rt_cnt++;
				}

				rdmPOMDP.setInitialBelief(initialbelief);
				RDMSimConnector.p = rdmPOMDP;

				rdmConnector.performAction(selectedAction);

				prevState = cs;
				prevAction = selectedAction;

				System.out.println("Topology Count:: MST: "+mst_cnt+" RT: "+rt_cnt);
			}

			// Per-NFR run summary (see writeRunSummary).
			writeRunSummary(outputDir, observedSteps, stateChanges, mst_cnt, rt_cnt,
				mon1SatCount, mon2SatCount, mon3SatCount,
				mon1ProbSum, mon2ProbSum, mon3ProbSum,
				activeLinksSum, bandwidthSum, timeToWriteSum, currentscenario_case);

			pw_mc_regr.flush();
			pw_mp_regr.flush();
			pw_mr_regr.flush();
			pw_mc_regr.close();
			pw_mr_regr.close();
			pw_mp_regr.close();

			// charts.html is ~5 MB per run, so the ablation driver turns it off with
			// -Dcharts=false: across the full grid it would cost several GB for plots nobody
			// opens. Cross-cell figures are built from experiments/results.csv instead.
			if ("false".equalsIgnoreCase(System.getProperty("charts", "true"))) {
				log.info("Chart generation skipped (-Dcharts=false)");
			} else {
				try {
					runPython(outputDir, mon1Threshold, mon2Threshold, mon3Threshold, lookback);
				} catch (Exception chartEx) {
					log.warn("Chart generation failed: {}", chartEx.getMessage());
				}
			}
		} catch (IOException ioex) {
			log.error("IOException in runCaseIoT", ioex);
		} catch (RuntimeException ex) {
			log.error("Unexpected exception in runCaseIoT", ex);
		}
	}

	/**
	 * Per-NFR run summary. Thresholds come from config_log_files/configuration.json.
	 * Satisfaction is reported per NFR rather than jointly: MON2 (bandwidth) and MON3
	 * (time to write) are both monotone increasing in active links, while MON1 requires
	 * active links to be high, so the three are not jointly satisfiable in general.
	 */
	private void writeRunSummary(String outputDir, int steps, int stateChanges, int mstCount, int rtCount,
		int mon1Sat, int mon2Sat, int mon3Sat,
		double mon1ProbSum, double mon2ProbSum, double mon3ProbSum,
		double activeLinksSum, double bandwidthSum, double timeToWriteSum, int scenario) {
		if (steps <= 0) {
			log.warn("No observed steps; skipping run summary");
			return;
		}

		int transitions = Math.max(0, steps - 1);
		double changeRate = transitions > 0 ? ((double) stateChanges) / transitions : 0.0;

		try (PrintWriter out = new PrintWriter(new FileWriter(new File(outputDir, "runSummary.txt")))) {
			out.printf("timesteps	%d%n", steps);
			out.printf("scenario	%d%n", scenario);
			out.printf("algorithm	%s%n", algorithmType);
			out.printf("lambda	%.4f%n", sp.getLambda());
			out.printf("runSeed	%d%n", runSeed);
			out.printf("surpriseMeasure	%s%n", surpriseMeasureForGamma);
			out.printf("p_c	%.4f%n", p_c);
			out.printf("lookback	%d%n", lookback);
			out.printf("useSurpriseUpdating	%b%n", useSurpriseUpdating);
			out.printf("priorStrength	%.4f%n", PomdpParser.priorStrength);
			out.println();
			out.printf("MON1_satisfaction_rate	%.4f%n", ((double) mon1Sat) / steps);
			out.printf("MON2_satisfaction_rate	%.4f%n", ((double) mon2Sat) / steps);
			out.printf("MON3_satisfaction_rate	%.4f%n", ((double) mon3Sat) / steps);
			out.printf("MON1_mean_belief_prob	%.4f%n", mon1ProbSum / steps);
			out.printf("MON2_mean_belief_prob	%.4f%n", mon2ProbSum / steps);
			out.printf("MON3_mean_belief_prob	%.4f%n", mon3ProbSum / steps);
			out.println();
			out.printf("mean_active_links	%.2f%n", activeLinksSum / steps);
			out.printf("mean_bandwidth_consumption	%.2f%n", bandwidthSum / steps);
			out.printf("mean_time_to_write	%.2f%n", timeToWriteSum / steps);
			out.printf("threshold_active_links	%.2f%n", mon1Threshold);
			out.printf("threshold_bandwidth_consumption	%.2f%n", mon2Threshold);
			out.printf("threshold_time_to_write	%.2f%n", mon3Threshold);
			out.println();
			out.printf("action_MST	%d%n", mstCount);
			out.printf("action_RT	%d%n", rtCount);
			out.printf("state_changes	%d%n", stateChanges);
			out.printf("transitions_recorded	%d%n", transitions);
			out.printf("state_change_rate	%.4f%n", changeRate);
		} catch (IOException ex) {
			log.warn("Could not write run summary: {}", ex.getMessage());
		}

		log.info("Per-NFR satisfaction: MON1={}%, MON2={}%, MON3={}%",
			String.format("%.1f", 100.0 * mon1Sat / steps),
			String.format("%.1f", 100.0 * mon2Sat / steps),
			String.format("%.1f", 100.0 * mon3Sat / steps));
		log.info("Actions: MST={} RT={}; state changes {}/{} ({}%)", mstCount, rtCount, stateChanges,
			transitions, String.format("%.1f", 100.0 * changeRate));
	}

	/**
	 * Solve a POMDP defined by a .POMDP file
	 * 
	 * @param pomdpFileName filename of a domain in the domain directory
	 */
	public void run(String pomdpFileName) {
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
		System.out.println("Running RDMSim POMDP simulation...");
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
		pomdp.run("RDM.POMDP");

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
