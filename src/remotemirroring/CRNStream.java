package remotemirroring;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import rdm.network.Monitorables;
import rdm.network.Topology;
import rdm.network.TopologyList;
import rdm.simulation.UncertaintyScenario;

/**
 * Common random numbers for the ablation grid.
 *
 * <p>RDMSim generates its monitorables with {@code ThreadLocalRandom}, which cannot be
 * seeded from outside the jar. Two ablation arms in the same scenario therefore face
 * different environment realisations, and an observed difference between them is
 * indistinguishable from noise without a large number of repeats.
 *
 * <p>This class pre-draws the environment for every timestep, <b>for both topologies</b>,
 * from a {@link Random} seeded by {@code (scenario, seed)}. Every arm in a
 * {@code (scenario, seed)} cell then consumes the same stream, indexed by
 * {@code (timestep, topology)} - so arms are perfectly paired and differ only where their
 * action paths diverge. Because both topologies are drawn at every timestep, an arm that
 * chooses MST at t does not shift the numbers a different arm sees for RT at t.
 *
 * <p>The draw and the arithmetic replicate
 * {@code rdm.network.Monitorables.generateMonitorables} exactly:
 * <pre>
 *   dev_*         ~ U[min, max+1)   as a double percentage, with 0 &lt; dev &lt; 1 clamped to 0
 *   raw_links     ~ randint[topology.minActiveLinks, topology.maxActiveLinks]
 *   active_links  = (int)(raw_links - (dev_links/100) * raw_links)
 *   bandwidth     = alpha * active_links * randint[20,30];  bandwidth += (dev_bw/100) * bandwidth
 *   time_to_write = alpha * active_links * randint[10,20];  ttw += (dev_ttw/100) * ttw
 * </pre>
 * with {@code alpha = 1.0} as the jar hardcodes it. {@code CRNVerify} asserts that the
 * resulting marginal NFR satisfaction rates match the analytic truth table, which is the
 * acceptance test for this re-implementation.
 *
 * <p>Thresholds are never touched: those still come from configuration.json via RDMSim.
 */
public final class CRNStream {
	private static final Logger log = LogManager.getLogger(CRNStream.class);

	/** The jar hardcodes alpha to 1.0 on every generateMonitorables call. */
	public static final double ALPHA = 1.0;

	/** One timestep's environment for one topology. */
	public static final class Draw {
		public final int activeLinks;
		public final double bandwidthConsumption;
		public final double timeToWrite;

		Draw(int activeLinks, double bandwidthConsumption, double timeToWrite) {
			this.activeLinks = activeLinks;
			this.bandwidthConsumption = bandwidthConsumption;
			this.timeToWrite = timeToWrite;
		}
	}

	/** Deviation bounds and link range for one topology. */
	private static final class TopologySpec {
		final String name;
		final int minLinks;
		final int maxLinks;
		final int minDevLinks, maxDevLinks;
		final int minDevBandwidth, maxDevBandwidth;
		final int minDevWritingTime, maxDevWritingTime;

		TopologySpec(String name, int minLinks, int maxLinks, int minDevLinks, int maxDevLinks,
				int minDevBandwidth, int maxDevBandwidth, int minDevWritingTime, int maxDevWritingTime) {
			this.name = name;
			this.minLinks = minLinks;
			this.maxLinks = maxLinks;
			this.minDevLinks = minDevLinks;
			this.maxDevLinks = maxDevLinks;
			this.minDevBandwidth = minDevBandwidth;
			this.maxDevBandwidth = maxDevBandwidth;
			this.minDevWritingTime = minDevWritingTime;
			this.maxDevWritingTime = maxDevWritingTime;
		}
	}

	private final int scenario;
	private final int seed;
	private final int horizon;
	/** Topology name to per-timestep draws. Insertion-ordered so the draw order is fixed. */
	private final Map<String, Draw[]> draws = new LinkedHashMap<>();

	/**
	 * Pre-draw the whole run.
	 *
	 * @param scenario the uncertainty scenario in force (0..6), for the seed and for logging
	 * @param seed     the run seed; cells sharing (scenario, seed) share this environment
	 * @param horizon  number of timesteps to pre-draw
	 * @param us       the scenario deviation bounds, after any -Dscenario override
	 */
	public CRNStream(int scenario, int seed, int horizon, UncertaintyScenario us) {
		if (horizon <= 0) {
			throw new IllegalArgumentException("horizon must be > 0; got " + horizon);
		}
		this.scenario = scenario;
		this.seed = seed;
		this.horizon = horizon;

		ArrayList<TopologySpec> specs = buildSpecs(us);
		// One stream for the whole run. Drawing every topology at every timestep in a fixed
		// order is what makes the stream independent of the action path an arm takes.
		Random rng = new Random(streamSeed(scenario, seed));
		for (TopologySpec spec : specs) {
			draws.put(spec.name, new Draw[horizon]);
		}
		for (int t = 0; t < horizon; t++) {
			for (TopologySpec spec : specs) {
				draws.get(spec.name)[t] = drawOne(rng, spec);
			}
		}

		log.info("CRN stream: scenario={}, seed={}, horizon={}, topologies={}", scenario, seed, horizon,
				draws.keySet());
	}

	/**
	 * Mix the two so neighbouring seeds do not produce correlated streams.
	 */
	public static long streamSeed(int scenario, int seed) {
		return ((long) seed) * 1000003L + ((long) scenario) * 7919L;
	}

	/**
	 * Read each topology link range from RDMSim's loaded topology list, and pair it with
	 * the matching per-topology deviation bounds from the scenario.
	 */
	private static ArrayList<TopologySpec> buildSpecs(UncertaintyScenario us) {
		if (TopologyList.topologies == null || TopologyList.topologies.isEmpty()) {
			throw new RuntimeException("TopologyList.topologies is empty; RDMSim was not initialised");
		}
		ArrayList<TopologySpec> specs = new ArrayList<>();
		for (Topology topology : TopologyList.topologies) {
			String name = topology.getTopologyName();
			int minLinks = (int) topology.getMinActiveLinks();
			int maxLinks = (int) topology.getMaxActiveLinks();
			if (maxLinks < minLinks) {
				throw new RuntimeException("Topology '" + name + "' has maxActiveLinks < minActiveLinks");
			}
			boolean isMst = "mst".equalsIgnoreCase(name);
			specs.add(new TopologySpec(name, minLinks, maxLinks,
					isMst ? us.getMinDeviationMSTLinks() : us.getMinDeviationRTLinks(),
					isMst ? us.getMaxDeviationMSTLinks() : us.getMaxDeviationRTLinks(),
					isMst ? us.getMinDeviationMSTBandwidth() : us.getMinDeviationRTBandwidth(),
					isMst ? us.getMaxDeviationMSTBandwidth() : us.getMaxDeviationRTBandwidth(),
					isMst ? us.getMinDeviationMSTWritingTime() : us.getMinDeviationRTWritingTime(),
					isMst ? us.getMaxDeviationMSTWritingTime() : us.getMaxDeviationRTWritingTime()));
		}
		return specs;
	}

	private static Draw drawOne(Random rng, TopologySpec spec) {
		double devLinks = deviation(rng, spec.minDevLinks, spec.maxDevLinks);
		double devBandwidth = deviation(rng, spec.minDevBandwidth, spec.maxDevBandwidth);
		double devWritingTime = deviation(rng, spec.minDevWritingTime, spec.maxDevWritingTime);

		int rawLinks = randomInt(rng, spec.minLinks, spec.maxLinks);
		int activeLinks = (int) (rawLinks - (devLinks / 100.0) * rawLinks);

		double bandwidth = ALPHA * activeLinks * randomInt(rng, 20, 30);
		bandwidth += (devBandwidth / 100.0) * bandwidth;

		double timeToWrite = ALPHA * activeLinks * randomInt(rng, 10, 20);
		timeToWrite += (devWritingTime / 100.0) * timeToWrite;

		return new Draw(activeLinks, bandwidth, timeToWrite);
	}

	/**
	 * The jar draws the deviation as {@code nextDouble(min, max + 1)} and then clamps any
	 * value in the open interval (0, 1) to zero. With bounds [0,0] that clamp is what makes
	 * an undeviated topology exactly undeviated.
	 */
	private static double deviation(Random rng, int min, int max) {
		double value = min + rng.nextDouble() * ((max + 1) - min);
		return (value > 0 && value < 1) ? 0.0 : value;
	}

	/** Inclusive on both ends, matching {@code nextInt(min, max + 1)}. */
	private static int randomInt(Random rng, int min, int max) {
		return min + rng.nextInt(max - min + 1);
	}

	/**
	 * The pre-drawn environment for a timestep and topology.
	 *
	 * @throws RuntimeException if the timestep is past the pre-drawn horizon, which would
	 *                          silently break pairing rather than fail loudly
	 */
	public Draw get(int timestep, String topologyName) {
		Draw[] series = draws.get(topologyName);
		if (series == null) {
			// Try a case-insensitive match before giving up; RDMSim is inconsistent about case.
			for (Map.Entry<String, Draw[]> entry : draws.entrySet()) {
				if (entry.getKey().equalsIgnoreCase(topologyName)) {
					series = entry.getValue();
					break;
				}
			}
		}
		if (series == null) {
			throw new RuntimeException("No CRN stream for topology '" + topologyName + "'; have " + draws.keySet());
		}
		if (timestep < 0 || timestep >= horizon) {
			throw new RuntimeException("CRN stream exhausted: timestep " + timestep + " outside [0, " + horizon
					+ "). Pre-draw the stream with the same horizon the run uses.");
		}
		return series[timestep];
	}

	/**
	 * Overwrite the monitorables RDMSim just generated with the pre-drawn values for this
	 * timestep and topology.
	 *
	 * <p>Both the {@link Monitorables} object and the probe's own fields are written, because
	 * {@code POMDP.getInitialStateRDM()} and {@code POMDP.getObservation()} read through the
	 * probe while the harness logs from the Monitorables object. Writing only one would leave
	 * the two disagreeing within a single timestep.
	 *
	 * @param timestep     the current timestep
	 * @param topologyName the topology whose environment is in force
	 * @param monitorables the object returned by {@code getMonitorables()}, mutated in place
	 */
	public void overwrite(int timestep, String topologyName, Monitorables monitorables) {
		Draw draw = get(timestep, topologyName);

		monitorables.setAlpha(ALPHA);
		monitorables.setActiveLinks(draw.activeLinks);
		monitorables.setBandwidthConsumption(draw.bandwidthConsumption);
		monitorables.setTimeToWrite(draw.timeToWrite);

		// The probe reads its own fields, not the Monitorables object's.
		RDMSimConnector.effector.setActiveLinks(draw.activeLinks);
		RDMSimConnector.effector.setBandwidthConsumption(draw.bandwidthConsumption);
		RDMSimConnector.effector.setTimeToWrite(draw.timeToWrite);
	}

	public int getScenario() {
		return scenario;
	}

	public int getSeed() {
		return seed;
	}

	public int getHorizon() {
		return horizon;
	}
}
