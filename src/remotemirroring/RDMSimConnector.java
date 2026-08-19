package remotemirroring;

import pomdp.POMDP;
import rdm.management.Effector;
import rdm.management.NetworkManagment;
import rdm.management.Probe;
import rdm.network.Monitorables;
import solver.BeliefPoint;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.special.Gamma;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * The connector between the Simulator and the MAPE-K loop
 */
public class RDMSimConnector {
	private static final Logger log = LogManager.getLogger(RDMSimConnector.class);

	/**
	 * Data class to hold MIP (Mutual Information Surprise) value along with its confidence bounds.
	 */
	public static class MIPResult {
		public final double mis;
		public final double lowerBound;
		public final double upperBound;
		
		public MIPResult(double mis, double lowerBound, double upperBound) {
			this.mis = mis;
			this.lowerBound = lowerBound;
			this.upperBound = upperBound;
		}
		
		/**
		 * Returns a MIPResult with all values set to 0.0 (used when insufficient history).
		 */
		public static MIPResult zero() {
			return new MIPResult(0.0, 0.0, 0.0);
		}
	}

	public static NetworkManagment network_management;
	public static boolean refsetcreation=false;
	public static Probe probe;
	public static Effector effector;
	public static int timestep;
	public static Monitorables monitorables;
	
	public static POMDP p;

	public int selectedindex;
	
	private double eps;
	
	// History of mutual information values per mote for MIP calculation
	private Map<Integer, ArrayList<Double>> miHistory;
	private int lookback = 4; // m - lookback period for MIP calculation (SMiLe via setLookback)
	// Track which timesteps have already had bounds written (to avoid duplicates)
	private int lastBoundsTimestep = -1;
	
	// Track if header has been written to mote_metrics.txt
	private static boolean moteMetricsHeaderWritten = false;
	
	// Surprise measure to use for gamma calculation: "CC" (Confidence-Corrected), "BF" (Bayes Factor), or "MIP" (Mutual Information Surprise)
	private String surpriseMeasureForGamma = "CC"; // Default to Confidence-Corrected Surprise
	
	// Probability of change (volatility): in (0, 1); controls m = p_c/(1-p_c) in SMiLe gamma formula. SMiLe for experiments.
	private double p_c = 0.5;
	
	// If true, use SMiLe (surprise-weighted) transition belief updates; if false, use classic Bayesian (Dirichlet +1 only).
	private boolean useSurpriseUpdating = true;
	
	// Output directory for file operations
	private String outputDirectory = "output_dir/"; // Default, will be set by SolvePOMDP
	
	
	public void setOutputDirectory(String outputDir) {
		this.outputDirectory = outputDir;
	}
	
	public String getOutputDirectory() {
		return this.outputDirectory;
	}
	
	/** Set p_c (probability of change) for SMiLe; must be in (0, 1). Used in gamma = m*S/(1+m*S) with m = p_c/(1-p_c). */
	public void setP_c(double p_c) {
		if (p_c <= 0 || p_c >= 1) {
			throw new IllegalArgumentException("p_c must be in (0, 1), got " + p_c);
		}
		this.p_c = p_c;
	}
	
	public double getP_c() {
		return this.p_c;
	}
	
	/** Set lookback period (m) for MIP calculation. Must be > 0. Used in MIP = MI[current] - MI[current - lookback]. */
	public void setLookback(int lookback) {
		if (lookback <= 0) {
			throw new IllegalArgumentException("lookback must be > 0, got " + lookback);
		}
		this.lookback = lookback;
	}
	
	public int getLookback() {
		return this.lookback;
	}
	
	/** Set whether to use surprise-based (SMiLe) or classic Bayesian transition belief updates. */
	public void setUseSurpriseUpdating(boolean useSurpriseUpdating) {
		this.useSurpriseUpdating = useSurpriseUpdating;
	}

	/**
	 * Set which surprise measure to use for gamma calculation
	 * @param measure "CC" for Confidence-Corrected Surprise, "BF" for Bayes Factor Surprise, or "MIP" for Mutual Information Surprise
	 */
	public void setSurpriseMeasureForGamma(String measure) {
		if (measure.equals("CC") || measure.equals("BF") || measure.equals("MIP")) {
			this.surpriseMeasureForGamma = measure;
		} else {
			log.warn("Invalid surprise measure '{}'; use CC, BF, or MIP. Keeping: {}", measure, this.surpriseMeasureForGamma);
		}
	}

	public String getSurpriseMeasureForGamma() {
		return this.surpriseMeasureForGamma;
	}

	public RDMSimConnector() {
		network_management=new NetworkManagment();
		probe=network_management.getProbe();
		effector=network_management.getEffector();
		selectedindex=0;
		eps = 1e-6; // prevent underflow
		miHistory = new HashMap<Integer, ArrayList<Double>>();
	}

	/**
	 * Calculates the Confidence-Corrected Surprise for a given action and observed next state.
	 * See "A taxonomy of surprise definitions" (2022): SCC1(yt+1|xt+1; pi(t)) := DKL[pi(t)||piflat(.|yt+1, xt+1)].
	 * For each possible current state, calculate KL divergence between the current transition belief
	 * (before update) and the flat prior updated with the new observation, weighted by the current
	 * state belief (to account for POMDP uncertainty over the current state).
	 */
	private double confidenceCorrectedSurprise(double[][][] transitionBelief, double[][][] transitionBeliefReset, int action, int nextstate) {
		int numStates = p.getNumStates();
		double surpriseCC = 0.0;

		for (int currState = 0; currState < numStates; currState++) {
			double[] alpha = transitionBelief[currState][action];
			double[] beta = transitionBeliefReset[currState][action];

			double a0 = 0.0;
			double b0 = 0.0;
			for (int i = 0; i < numStates; i++) {
				a0 += alpha[i];
				b0 += beta[i];
			}

			// https://statproofbook.github.io/P/dir-kl.html
			double term = Gamma.logGamma(a0) - Gamma.logGamma(b0);
			for (int i = 0; i < numStates; i++) {
				term += Gamma.logGamma(beta[i]) - Gamma.logGamma(alpha[i]);
			}
			for (int i = 0; i < numStates; i++) {
				double psiAlphaI = Gamma.digamma(alpha[i]);
				double psiAlpha0 = Gamma.digamma(a0);
				term += (alpha[i] - beta[i]) * (psiAlphaI - psiAlpha0);
			}
			surpriseCC += p.getInitialBelief().getBelief(currState) * term;
		}
		return surpriseCC;
	}

	/** Log predicted probabilities of the observed next state under the given transition belief, for each current state. */
	private double[] getLogPredProbs(double[][][] transitionBelief, int action, int nextstate) {
		double[] logPred = new double[p.getNumStates()];
		for (int stateIndex = 0; stateIndex < p.getNumStates(); stateIndex++) {
			double[] alpha = transitionBelief[stateIndex][action];
			double a0 = 0.0;
			for (double a : alpha) a0 += a;

			if (a0 < this.eps) a0 = this.eps;

			logPred[stateIndex] = Math.log(transitionBelief[stateIndex][action][nextstate]) - Math.log(a0);
		}
		return logPred;
	}

	/**
	 * Calculates the Bayes Factor Surprise for a given action and observed next state: the log ratio
	 * of the predicted probability of the observed next state under the flat (reset) prior versus the
	 * current belief, marginalized over current states weighted by the current belief.
	 */
	private double bayesFactorSurprise(double[][][] transitionBeliefCurr, double[][][] transitionBeliefReset, int action, int nextstate) {
		double[] logPredProbCurrVals = this.getLogPredProbs(transitionBeliefCurr, action, nextstate);
		double predProbCurr = 0;
		for (int currState = 0; currState < p.getNumStates(); currState++) {
			predProbCurr += p.getInitialBelief().getBelief(currState) * Math.exp(logPredProbCurrVals[currState]);
		}

		double[] logPredProbResetVals = this.getLogPredProbs(transitionBeliefReset, action, nextstate);
		double predProbReset = 0;
		for (int currState = 0; currState < p.getNumStates(); currState++) {
			predProbReset += p.getInitialBelief().getBelief(currState) * Math.exp(logPredProbResetVals[currState]);
		}

		return Math.log(Math.max(this.eps, predProbReset)) - Math.log(Math.max(this.eps, predProbCurr));
	}

	/** Entropy of one Dirichlet distribution of alpha pseudo-counts. */
	private double dirichlet_entropy(double[] alpha) {
		double alpha0 = Arrays.stream(alpha).sum();
		int k = alpha.length;

		double lnB = 0.0;
		for (double a : alpha) {
			lnB += Gamma.logGamma(a);
		}
		lnB -= Gamma.logGamma(alpha0);

		double sum1 = 0.0;
		for (double a : alpha) {
			sum1 += (a - 1.0) * Gamma.digamma(a);
		}

		double sum2 = (alpha0 - k) * Gamma.digamma(alpha0);

		return (lnB + sum2 - sum1);
	}

	/** Expected entropy of the network's transition belief for a given action and next state, weighted by current state belief. */
	private double getEntropy(double[][][] transitionBelief, int action, int nextstate) {
		double entropy = 0.0;
		for (int stateIndex = 0; stateIndex < p.getNumStates(); stateIndex++) {
			entropy += p.getInitialBelief().getBelief(stateIndex) * dirichlet_entropy(transitionBelief[stateIndex][action]);
		}
		return entropy;
	}

	private void appendMIPBoundsToFile(int timestep, double lowerBound, double upperBound) {
		new OutputLogger(outputDirectory).appendRow("MIPBounds.txt", timestep, lowerBound, upperBound);
	}

	/**
	 * Calculates Mutual Information Surprise (MIP) for the current timestep: the difference in mutual
	 * information (MI) between now and "lookback" timesteps earlier. Returns MIPResult with MIP value
	 * and confidence bounds; if there isn't enough MI history yet, returns MIPResult.zero().
	 */
	private MIPResult calculateAndStoreMIP(double[][][] transitionBeliefPrior, double[][][] transitionBeliefPosterior, int action, int nextstate, int timestep) {
		double priorEntropy = this.getEntropy(transitionBeliefPrior, action, nextstate);
		double posteriorEntropy = this.getEntropy(transitionBeliefPosterior, action, nextstate);

		double mutualInformation = priorEntropy - posteriorEntropy;

		// Single running MI history (RDMSim has one network per timestep, unlike DeltaIoT's per-mote history)
		if (!miHistory.containsKey(0)) {
			miHistory.put(0, new ArrayList<Double>());
		}
		ArrayList<Double> history = miHistory.get(0);
		history.add(mutualInformation);

		if (history.size() > lookback) {
			double mis = history.get(history.size() - 1) - history.get(history.size() - 1 - lookback);

			// Theorem 1: Î_{n+m} - Î_n in (log(m + n) - log n) +/- (2m log(2/rho) log(m + n)) / (m + n)
			double rho = 0.05; // confidence level (0.05 -> 95% confidence)
			int n = history.size() - lookback;
			int m = lookback;
			int nPlusM = history.size();

			double pivotVal = Math.log(nPlusM) - Math.log(n);
			double errorTerm = Math.sqrt((2.0 * m * Math.log(2.0 / rho))) * Math.log(nPlusM) / nPlusM;

			double upperBound = pivotVal + errorTerm;
			double lowerBound = pivotVal - errorTerm;

			if (timestep > lastBoundsTimestep) {
				appendMIPBoundsToFile(timestep, lowerBound, upperBound);
				lastBoundsTimestep = timestep;
			}

			return new MIPResult(mis, lowerBound, upperBound);
		}

		return MIPResult.zero();
	}

	private void updateObservationBelief(int action, int nextstate, int obs) {
		p.observationBelief[action][nextstate][obs] += 1.0;
	}

	private void updateTransitionBelief(int action, int nextstate) {
		double[][][] transitionBeliefCurrTemp = Arrays.stream(p.transitionBeliefCurr)
				.map(twoD -> Arrays.stream(twoD).map(arr -> arr.clone()).toArray(double[][]::new))
				.toArray(double[][][]::new);
		double[][][] transitionBeliefResetTemp = Arrays.stream(p.transitionBeliefReset)
				.map(twoD -> Arrays.stream(twoD).map(arr -> arr.clone()).toArray(double[][]::new))
				.toArray(double[][][]::new);

		for (int stateIndex = 0; stateIndex < p.getNumStates(); stateIndex++) {
			transitionBeliefCurrTemp[stateIndex][action][nextstate] += 1.0;
			transitionBeliefResetTemp[stateIndex][action][nextstate] += 1.0;
		}

		double surpriseCC = confidenceCorrectedSurprise(p.transitionBeliefCurr, transitionBeliefResetTemp, action, nextstate);
		double logSurpriseCC = Math.log(Math.max(this.eps, surpriseCC));
		double logSurpriseBF = Math.max(this.eps, bayesFactorSurprise(p.transitionBeliefCurr, p.transitionBeliefReset, action, nextstate));
		MIPResult misResult = calculateAndStoreMIP(p.transitionBeliefCurr, transitionBeliefCurrTemp, action, nextstate, RDMSimConnector.timestep);
		double currentMIP = misResult.mis;

		double logSurprise = 0.0;
		if (surpriseMeasureForGamma.equals("CC")) {
			logSurprise = logSurpriseCC;
		} else if (surpriseMeasureForGamma.equals("BF")) {
			logSurprise = logSurpriseBF;
		} else if (surpriseMeasureForGamma.equals("MIP")) {
			double absMIP = Math.abs(currentMIP);
			double scaledMIP = Math.max(this.eps, absMIP);
			logSurprise = Math.log(scaledMIP);
		}

		OutputLogger outLog = new OutputLogger(outputDirectory);

		if (!useSurpriseUpdating) {
			p.transitionBeliefCurr = transitionBeliefCurrTemp;
			outLog.appendRow("surpriseBF.txt", RDMSimConnector.timestep, Math.exp(logSurpriseBF));
			outLog.appendRow("gamma.txt", RDMSimConnector.timestep, 0);
			outLog.appendRow("surpriseCC.txt", RDMSimConnector.timestep, Math.exp(logSurpriseCC));
			outLog.appendRow("surpriseMIP.txt", RDMSimConnector.timestep, currentMIP);
			return;
		}

		// SMiLe gamma formula (Definition 4): gamma(S, m) = mS / (1 + mS), m = p_c/(1-p_c)
		double m = this.p_c / (1.0 - this.p_c);
		double gamma = 1.0 / (1.0 + (1 / (m * Math.exp(logSurprise))));
		gamma = Math.max(this.eps, gamma);

		outLog.appendRow("surpriseBF.txt", RDMSimConnector.timestep, Math.exp(logSurpriseBF));
		outLog.appendRow("gamma.txt", RDMSimConnector.timestep, gamma);
		outLog.appendRow("surpriseCC.txt", RDMSimConnector.timestep, Math.exp(logSurpriseCC));
		outLog.appendRow("surpriseMIP.txt", RDMSimConnector.timestep, currentMIP);

		// SMiLe updating of transitionBeliefCurr: new_belief = (1-gamma)*updated_current + gamma*updated_flat_prior
		for (int stateIndex = 0; stateIndex < p.getNumStates(); stateIndex++) {
			for (int nextStateIndex = 0; nextStateIndex < p.getNumStates(); nextStateIndex++) {
				transitionBeliefCurrTemp[stateIndex][action][nextStateIndex] =
						(1 - gamma) * transitionBeliefCurrTemp[stateIndex][action][nextStateIndex]
						+ gamma * transitionBeliefResetTemp[stateIndex][action][nextStateIndex];
			}
		}

		p.transitionBeliefCurr = transitionBeliefCurrTemp;
	}
	
	public int performAction(int selectedaction)
	{
		
	////Perform ITP or DTP on the link on the simulator
			///return rewards and observations
			//update belief value and change initial belief
			
			///Immediate Reward
			double r=p.getReward(p.getCurrentState(), selectedaction);
			int nextstate;
			if(selectedaction==0)
			{
				
				
				nextstate=p.nextStateRDM(p.getCurrentState(), selectedaction);
				p.setCurrentState(nextstate);
				
			}
			else {

				nextstate=p.nextStateRDM(p.getCurrentState(), selectedaction);
				p.setCurrentState(nextstate);
			}

			/// Update transition beliefs (CC/BF/MIP surprise + SMiLe gamma) based on the new observation
			this.updateTransitionBelief(selectedaction, nextstate);

			///Observation
			int obs=p.getObservation(selectedaction, nextstate);
			this.updateObservationBelief(selectedaction, nextstate, obs);
			BeliefPoint b=p.updateBelief(p.getInitialBelief(), selectedaction, obs);
			p.setInitialBelief(b);
			
			//p.getReward(s, action);
			
			/*S currentS  = states.stateIdentifier(currentState);
			
			S nextState = this.transitions.nextState(currentS, action);
			
			this.currentState = states.stateNumber(nextState);
			
			
			double[] reward = this.rewards.getReward(currentS, action, nextState);
			
			
			
				O obs = this.observationFunction.getObservation(action, nextState);
				
				this.beliefUpdate(action, obs);*/
		
			
			
			return 0;

			
		
	}
	
	

}
