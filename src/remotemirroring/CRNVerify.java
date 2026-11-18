package remotemirroring;

import rdm.management.NetworkManagment;
import rdm.network.Monitorables;
import rdm.network.NetworkProperties;
import rdm.simulation.SimulationProperties;

/**
 * Acceptance test for {@link CRNStream}: does the re-implemented generator reproduce
 * RDMSim's own marginal distributions?
 *
 * <p>{@code CRNStream} re-implements the arithmetic inside
 * {@code Monitorables.generateMonitorables}, which is the one piece of this work that could
 * silently change the exemplar's behaviour. The check here is independent of that
 * re-implementation: the expected rates were derived analytically from the published
 * generative formula (uniform link draws, uniform 20-30 and 10-20 multipliers, percentage
 * deviations) by Monte-Carlo integration outside this codebase. If CRNStream drifts from the
 * jar, these numbers stop matching.
 *
 * <p>Run from the project root, after a build:
 * <pre>
 *   java -cp "bin;libraries/*" remotemirroring.CRNVerify
 * </pre>
 * Exits non-zero if any rate is outside tolerance, so it can gate the grid in CI or a script.
 */
public final class CRNVerify {

	/** Draws per (scenario, topology). Monte-Carlo error at this size is about 0.0008. */
	private static final int DRAWS = 400_000;

	/**
	 * Tolerance per rate. Both sides of the comparison are Monte-Carlo estimates (the
	 * expected values to about 0.002, these draws to about 0.0008), so anything inside 0.005
	 * is agreement and anything outside it is a real divergence.
	 */
	private static final double TOLERANCE = 0.005;

	/**
	 * Expected {MON1, MON2, MON3, joint} per scenario, MST then RT. Derived analytically
	 * from the generative formula; see the ground-truth table in docs/ABLATION.md.
	 */
	private static final double[][][] EXPECTED = {
			// scenario 0
			{ { 0.464, 0.811, 0.929, 0.254 }, { 1.000, 0.068, 0.393, 0.045 } },
			// scenario 1
			{ { 0.374, 0.903, 0.976, 0.266 }, { 1.000, 0.069, 0.394, 0.046 } },
			// scenario 2
			{ { 0.464, 0.811, 0.929, 0.256 }, { 1.000, 0.016, 0.269, 0.009 } },
			// scenario 3
			{ { 0.374, 0.903, 0.976, 0.266 }, { 1.000, 0.016, 0.269, 0.009 } },
			// scenario 4
			{ { 0.374, 0.823, 0.937, 0.178 }, { 1.000, 0.068, 0.392, 0.045 } },
			// scenario 5
			{ { 0.464, 0.812, 0.929, 0.256 }, { 1.000, 0.078, 0.413, 0.053 } },
			// scenario 6
			{ { 0.373, 0.824, 0.936, 0.177 }, { 1.000, 0.079, 0.414, 0.055 } },
	};

	private static final String[] TOPOLOGIES = { "mst", "rt" };

	private CRNVerify() {
	}

	public static void main(String[] args) {
		// Constructing NetworkManagment loads configuration.json: topology link ranges, the
		// NFR thresholds, and the initial uncertainty scenario.
		new NetworkManagment();
		SimulationProperties simulationProperties = NetworkManagment.simulation_properties;
		Monitorables m = NetworkProperties.m;

		double mon1Threshold = m.getThresholdActiveLinks();
		double mon2Threshold = m.getThresholdBandwidthConsumption();
		double mon3Threshold = m.getThresholdTimeToWrite();
		if (mon1Threshold <= 0 || mon2Threshold <= 0 || mon3Threshold <= 0) {
			System.err.println("NFR thresholds not loaded; run from the project root. cwd="
					+ System.getProperty("user.dir"));
			System.exit(2);
		}

		System.out.printf("CRNStream verification: %d draws per cell, tolerance %.3f%n", DRAWS, TOLERANCE);
		System.out.printf("Thresholds: MON1 >= %.0f, MON2 <= %.0f, MON3 <= %.0f%n%n",
				mon1Threshold, mon2Threshold, mon3Threshold);
		System.out.println("scen act   MON1   MON2   MON3  JOINT     (expected)            worst");

		int failures = 0;
		for (int scenario = 0; scenario < 7; scenario++) {
			ScenarioOverride.applyScenario(simulationProperties, scenario);
			// A fresh stream per scenario. Seed is arbitrary here: we are checking the
			// distribution, not a particular realisation.
			CRNStream stream = new CRNStream(scenario, 222, DRAWS,
					simulationProperties.getUncertaintyScenario());

			for (int topologyIndex = 0; topologyIndex < TOPOLOGIES.length; topologyIndex++) {
				String topology = TOPOLOGIES[topologyIndex];
				int mon1 = 0, mon2 = 0, mon3 = 0, joint = 0;
				for (int t = 0; t < DRAWS; t++) {
					CRNStream.Draw draw = stream.get(t, topology);
					boolean sat1 = draw.activeLinks >= mon1Threshold;
					boolean sat2 = draw.bandwidthConsumption <= mon2Threshold;
					boolean sat3 = draw.timeToWrite <= mon3Threshold;
					if (sat1) {
						mon1++;
					}
					if (sat2) {
						mon2++;
					}
					if (sat3) {
						mon3++;
					}
					if (sat1 && sat2 && sat3) {
						joint++;
					}
				}

				double[] actual = { rate(mon1), rate(mon2), rate(mon3), rate(joint) };
				double[] expected = EXPECTED[scenario][topologyIndex];
				double worst = 0.0;
				for (int i = 0; i < 4; i++) {
					worst = Math.max(worst, Math.abs(actual[i] - expected[i]));
				}
				boolean ok = worst <= TOLERANCE;
				if (!ok) {
					failures++;
				}

				System.out.printf("S%d  %-4s %.3f  %.3f  %.3f  %.3f   (%.3f %.3f %.3f %.3f)  %.4f %s%n",
						scenario, topology, actual[0], actual[1], actual[2], actual[3],
						expected[0], expected[1], expected[2], expected[3], worst, ok ? "ok" : "FAIL");
			}
		}

		System.out.println();
		if (failures == 0) {
			System.out.println("PASS: CRNStream reproduces the analytic marginals in all 14 cells.");
			System.exit(0);
		}
		System.out.printf("FAIL: %d of 14 cells outside tolerance. CRNStream does not match the%n", failures);
		System.out.println("      jar's generator - do not run the grid until this is resolved.");
		System.exit(1);
	}

	private static double rate(int count) {
		return ((double) count) / DRAWS;
	}
}
