package remotemirroring;

import pomdp.POMDP;
import rdm.management.Effector;
import rdm.management.NetworkManagment;
import rdm.management.Probe;
import rdm.network.Monitorables;
import solver.BeliefPoint;

import java.util.ArrayList;
import java.util.Map;

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
	
	public RDMSimConnector() {
		network_management=new NetworkManagment();
		probe=network_management.getProbe();
		effector=network_management.getEffector();
		selectedindex=0;
		eps = 1e-6; // prevent underflow
		miHistory = new HashMap<Integer, ArrayList<Double>>();
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
			
			///Observation
			int obs=p.getObservation(selectedaction, nextstate);
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
