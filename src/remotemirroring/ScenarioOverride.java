package remotemirroring;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import rdm.simulation.SimulationProperties;
import rdm.simulation.UncertaintyScenario;

/**
 * Runtime overrides for the uncertainty scenario and the simulation horizon.
 *
 * <p>The ablation driver runs many cells concurrently from a single working directory, so it
 * cannot edit {@code config_log_files/configuration.json} per cell - that file is shared
 * global state and concurrent writers would corrupt each other. Instead the driver passes
 * {@code -Dscenario=N} and {@code -Dtimesteps=T}, and this class applies them to the already
 * constructed {@link SimulationProperties} via RDMSim's public setters.
 *
 * <p>The deviation bounds for scenario N are read from the same
 * {@code deviation_scenario_N_*} keys that {@code SimulationProperties.loadSimulationSettings()}
 * would have used, so an overridden run is identical to one configured through the JSON file.
 * Parsing is done with a regex rather than a JSON library because two incompatible
 * json-simple versions sit on the classpath and the keys of interest are flat
 * {@code "name": [min, max]} pairs.
 *
 * <p>The existing {@link UncertaintyScenario} is mutated in place rather than replaced,
 * because RDMSim holds references to it from both
 * {@code SimulationProperties.getUncertaintyScenario()} and the static
 * {@code SimulationProperties.current_scenario}; mutating covers every holder.
 */
public final class ScenarioOverride {
	private static final Logger log = LogManager.getLogger(ScenarioOverride.class);

	/** Where the deviation bounds live. Relative, like every other path in this project. */
	private static final String CONFIG_PATH = "config_log_files/configuration.json";

	private ScenarioOverride() {
	}

	/**
	 * Apply {@code -Dscenario} and {@code -Dtimesteps} if present. Safe to call when neither
	 * is set: it then does nothing and the values from configuration.json stand.
	 *
	 * @param simulationProperties the properties object RDMSim has already populated
	 * @return the scenario in force after any override, for logging and the run summary
	 */
	public static int apply(SimulationProperties simulationProperties) {
		UncertaintyScenario us = simulationProperties.getUncertaintyScenario();
		if (us == null) {
			throw new RuntimeException("UncertaintyScenario is null; RDMSim was not initialised");
		}

		String scenarioProperty = System.getProperty("scenario");
		if (scenarioProperty != null && !scenarioProperty.trim().isEmpty()) {
			int scenario = parseInt(scenarioProperty.trim(), "scenario");
			if (scenario < 0 || scenario > 6) {
				throw new RuntimeException("scenario must be in 0..6; got " + scenario);
			}
			applyScenario(us, scenario);
			// Keep both holders consistent, in case a future RDMSim build reads one and not
			// the other.
			simulationProperties.setUncertaintyScenario(us);
			log.info("Scenario overridden to {} by -Dscenario", scenario);
		}

		String timestepsProperty = System.getProperty("timesteps");
		if (timestepsProperty != null && !timestepsProperty.trim().isEmpty()) {
			int timesteps = parseInt(timestepsProperty.trim(), "timesteps");
			if (timesteps <= 0) {
				throw new RuntimeException("timesteps must be > 0; got " + timesteps);
			}
			simulationProperties.setSimulationRuns(timesteps);
			log.info("Horizon overridden to {} timesteps by -Dtimesteps", timesteps);
		}

		return us.getCurrentScenario();
	}

	/**
	 * Set the t=0 topology deterministically, following RDMSim's own per-scenario rule.
	 *
	 * <p>RDMSim chooses the initial topology inside {@code loadSimulationSettings()}, at load
	 * time, from the JSON's {@code current_scenario}: MST for 0/1/4, RT for 2/5, and a
	 * {@code ThreadLocalRandom} coin flip for 3/6. Two consequences matter for the ablation:
	 * a {@code -Dscenario} override arrives too late to change that choice, so every cell
	 * inherits the JSON scenario's rule; and with the JSON at scenario 3 the choice is an
	 * unseedable coin flip. Since the t=0 observation seeds the belief and the first action,
	 * a different start sends the whole trajectory elsewhere - so without this, identical
	 * cells were not reproducible and CRN-paired arms were only paired half the time.
	 *
	 * <p>For 3/6 the coin flip is kept (it is part of those scenarios' definition) but drawn
	 * from a Random seeded by (scenario, seed), so every arm in a cell starts the same way.
	 *
	 * @return the name of the topology now in force
	 */
	public static String applyInitialTopology(int scenario, int seed) {
		String name;
		switch (scenario) {
			case 0, 1, 4 -> name = "mst";
			case 2, 5 -> name = "rt";
			case 3, 6 -> name = new java.util.Random(CRNStream.streamSeed(scenario, seed) + 1L).nextBoolean()
					? "mst" : "rt";
			default -> throw new RuntimeException("scenario must be in 0..6; got " + scenario);
		}
		for (rdm.network.Topology topology : rdm.network.TopologyList.topologies) {
			if (topology.getTopologyName().equalsIgnoreCase(name)) {
				// setCurrentTopology, not setNetworkTopology: the latter also bumps the
				// simulation timestep and rewrites log.json.
				RDMSimConnector.effector.setCurrentTopology(topology);
				log.info("Initial topology for scenario {} (seed {}): {}", scenario, seed, name);
				return name;
			}
		}
		throw new RuntimeException("Topology '" + name + "' not found in TopologyList.topologies");
	}

	/**
	 * Apply a scenario explicitly, ignoring the {@code -Dscenario} property. Used by
	 * {@code CRNVerify}, which walks all seven scenarios in a single JVM.
	 */
	public static void applyScenario(SimulationProperties simulationProperties, int scenario) {
		if (scenario < 0 || scenario > 6) {
			throw new RuntimeException("scenario must be in 0..6; got " + scenario);
		}
		UncertaintyScenario us = simulationProperties.getUncertaintyScenario();
		if (us == null) {
			throw new RuntimeException("UncertaintyScenario is null; RDMSim was not initialised");
		}
		applyScenario(us, scenario);
		simulationProperties.setUncertaintyScenario(us);
	}

	/**
	 * Set the scenario index and its twelve deviation bounds from configuration.json.
	 */
	private static void applyScenario(UncertaintyScenario us, int scenario) {
		String json = readConfig();

		int[] mstLinks = readPair(json, scenario, "mst", "links");
		int[] mstBandwidth = readPair(json, scenario, "mst", "bandwidth_consumption");
		int[] mstWritingTime = readPair(json, scenario, "mst", "writing_time");
		int[] rtLinks = readPair(json, scenario, "rt", "links");
		int[] rtBandwidth = readPair(json, scenario, "rt", "bandwidth_consumption");
		int[] rtWritingTime = readPair(json, scenario, "rt", "writing_time");

		us.setCurrentScenario(scenario);
		us.setMinDeviationMSTLinks(mstLinks[0]);
		us.setMaxDeviationMSTLinks(mstLinks[1]);
		us.setMinDeviationMSTBandwidth(mstBandwidth[0]);
		us.setMaxDeviationMSTBandwidth(mstBandwidth[1]);
		us.setMinDeviationMSTWritingTime(mstWritingTime[0]);
		us.setMaxDeviationMSTWritingTime(mstWritingTime[1]);
		us.setMinDeviationRTLinks(rtLinks[0]);
		us.setMaxDeviationRTLinks(rtLinks[1]);
		us.setMinDeviationRTBandwidth(rtBandwidth[0]);
		us.setMaxDeviationRTBandwidth(rtBandwidth[1]);
		us.setMinDeviationRTWritingTime(rtWritingTime[0]);
		us.setMaxDeviationRTWritingTime(rtWritingTime[1]);

		log.info("Scenario {} deviations: mst links{} bw{} ttw{}; rt links{} bw{} ttw{}", scenario,
				fmt(mstLinks), fmt(mstBandwidth), fmt(mstWritingTime),
				fmt(rtLinks), fmt(rtBandwidth), fmt(rtWritingTime));
	}

	private static String readConfig() {
		Path path = Paths.get(CONFIG_PATH);
		if (!Files.isReadable(path)) {
			throw new RuntimeException("Cannot read " + CONFIG_PATH
					+ "; run with the working directory set to the RDMSim_TAAS project root. cwd="
					+ System.getProperty("user.dir"));
		}
		try {
			return Files.readString(path);
		} catch (IOException ex) {
			throw new RuntimeException("Could not read " + CONFIG_PATH, ex);
		}
	}

	/**
	 * Extract a {@code "deviation_scenario_<n>_<topology>_<monitorable>": [min, max]} pair.
	 */
	private static int[] readPair(String json, int scenario, String topology, String monitorable) {
		String key = "deviation_scenario_" + scenario + "_" + topology + "_" + monitorable;
		Pattern pattern = Pattern.compile(
				"\"" + Pattern.quote(key) + "\"\\s*:\\s*\\[\\s*(-?\\d+)\\s*,\\s*(-?\\d+)\\s*\\]");
		Matcher matcher = pattern.matcher(json);
		if (!matcher.find()) {
			throw new RuntimeException("Key '" + key + "' not found in " + CONFIG_PATH);
		}
		int min = Integer.parseInt(matcher.group(1));
		int max = Integer.parseInt(matcher.group(2));
		if (max < min) {
			throw new RuntimeException("Key '" + key + "' has max < min: [" + min + ", " + max + "]");
		}
		return new int[] { min, max };
	}

	private static int parseInt(String value, String name) {
		try {
			return Integer.parseInt(value);
		} catch (NumberFormatException ex) {
			throw new RuntimeException("-D" + name + " must be an integer; got '" + value + "'");
		}
	}

	private static String fmt(int[] pair) {
		return "[" + pair[0] + "," + pair[1] + "]";
	}
}
