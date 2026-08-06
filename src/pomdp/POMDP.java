/*******************************************************************************
 * SolvePOMDP
 * Copyright (C) 2017 Erwin Walraven
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *******************************************************************************/

package pomdp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import deltaiot.services.Link;
import simulator.QoS;
import solver.BeliefPoint;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class POMDP {
	private static final Logger log = LogManager.getLogger(POMDP.class);	
	private String filename;
	private String instanceName;
	private int nStates;
	private int nActions;
	private int nObservations;
	private double discountFactor;
	private int currentState; ///Added for IoT
	
	private double[][] rewardFunction;
	//Changed transitionFunction to public from private
	public double[][][] transitionFunction;
	private double[][][] observationFunction;
	private double minReward = Double.POSITIVE_INFINITY;
	
	// alpha vectors for Dir(.) distribution representing transition probabilities
	public double[][][] transitionBeliefReset;
	public double[][][] transitionBeliefCurr;
	public double[][][] observationBelief;
	
	// vectors storing entropy at each timestep for transition belief distributions
	public double[] entropy;
	
	private BeliefPoint b0;
	
	private HashMap<Integer,String> actionLabels;
	
	/** NFR thresholds for state discretisation: MEC = energy consumption (default 20), RPL = packet loss ratio (default 0.2). */
	private double mecThreshold = 20.0;
	private double rplThreshold = 0.20;
	
	public POMDP(String filename, 
			int nStates, 
			int nActions, 
			int nObservations, 
			double discountFactor, 
			double[][] rewardFunction, 
			double[][][] transitionFunction, 
			double[][][] observationFunction, 
			HashMap<Integer,String> actionLabels, 
			BeliefPoint b0,
			double[][][] transitionBeliefReset, // effectively a collection of SxA dirichlet distribution hyperparameter collections of size S
			double[][][] transitionBeliefCurr,
			double [][][] observationBelief
			) {		
		// Extract just the filename, handling both Windows (\) and Unix (/) path separators
		String separator = filename.contains("\\") ? "\\\\" : "/";
		String[] filenameSplit = filename.split(separator);
		this.filename = filenameSplit[filenameSplit.length-1];
		// Remove .POMDP extension to get instance name
		this.instanceName = this.filename.replace(".POMDP", "").replace(".pomdp", "");
		this.nStates = nStates;
		this.nActions = nActions;
		this.nObservations = nObservations;
		this.discountFactor = discountFactor;
		this.rewardFunction = rewardFunction;
		this.transitionFunction = transitionFunction;
		this.observationFunction = observationFunction;
		this.actionLabels = actionLabels;
		this.b0 = b0;
				
		// Using beliefs instead of fixed probs for transitions
		this.transitionBeliefReset = transitionBeliefReset;
		this.transitionBeliefCurr = transitionBeliefCurr;
		
		// Also perform Bayesian updating for observation functions
		this.observationBelief = observationBelief;
		
		
		// compute min reward
		for(int s=0; s<nStates; s++) {
			for(int a=0; a<nActions; a++) {
				minReward = Math.min(minReward, rewardFunction[s][a]);
			}
		}
	}
	
	public int getNumStates() {
		return nStates;
	}
	
	public int getNumActions() {
		return nActions;
	}
	
	public int getNumObservations() {
		return nObservations;
	}
	
	public double getDiscountFactor() {
		return discountFactor;
	}
	
	public double getTransitionProbability(int s, int a, int sNext) {
		assert s < nStates && a < nActions && sNext < nStates;
		// take expectation over beliefs as an update of the world model
		double worldTransitionFn = transitionBeliefCurr[s][a][sNext] / (Arrays.stream(transitionBeliefCurr[s][a]).sum()); 
		return worldTransitionFn;
	}
		
	public double getReward(int s, int a) {
		assert s < nStates && a < nActions;
		return rewardFunction[s][a];
	}
	
	public double getObservationProbability(int a, int sNext, int o) {
		assert a < nActions && sNext<nStates && o < nObservations;
		// take expectation voer belifs as an update of the world model
		double worldObservationFn = observationBelief[a][sNext][o] / (Arrays.stream(observationBelief[a][sNext]).sum()); 
		return worldObservationFn;
	}
	
	public double getMinReward() {
		return minReward;
	}
	
	public String getFilename() {
		return filename;
	}
	
	public String getInstanceName() {
		return instanceName;
	}
	
	public String getActionLabel(int a) {
		return actionLabels.get(a);
	}
	
	/**
	 * 
	 * @param b = belief at current timestep
	 * @param a = action executed by SAS
	 * @param o = observation from action a
	 * @return
	 */
	public BeliefPoint updateBelief(BeliefPoint b, int a, int o) {
		assert a < nActions && o < nObservations;
		double[] newBelief = new double[nStates];
		
		// check if belief point has been prepared
		if(!b.hasActionObservationProbabilities()) {
			prepareBelief(b);
		}
		
		// compute normalizing constant
		double nc = b.getActionObservationProbability(a, o);
		assert nc > 0.0 : "o cannot be observed when executing a in belief b";
		
		// compute the new belief vector
		// -> For each state we are "possibly" in (according to belief likelihood), calculate transition probability for each state we could "possibly" end up in
		for(int sNext = 0; sNext < nStates; sNext++) {
			double beliefEntry = 0.0;
			
			for(int s = 0; s < nStates; s++) {
				// getTransitionProbability(s, a, sNext) is X_n (in  S!S!L)
				beliefEntry += getTransitionProbability(s, a, sNext) * b.getBelief(s);
			}
			
			newBelief[sNext] = beliefEntry * (getObservationProbability(a, sNext, o) / nc);
		}
		
		return new BeliefPoint(newBelief);
	}
	
	/**
	 * Calculates the `aoprobs` for the belief, b. 
	 * This is done by iterating over the matrix and for each possible state, calculate the transition probability (getTransitionProbability()).
	 * Then sum the average probability based on confidence across all belief state probabilities.
	 * @param b
	 */
	public void prepareBelief(BeliefPoint b) {
		assert b != null;
		if(b.hasActionObservationProbabilities()) return;
		
		double[][] aoProbs = new double[nActions][nObservations];
		
		for(int action = 0; action < nActions; action++) {
			for(int obs=0; obs < nObservations; obs++) {
				double prob = 0.0;
				
				for(int sNext=0; sNext < nStates; sNext++) {
					double p = 0.0;
					
					for(int s=0; s<nStates; s++) {
						// p = the belief-confidence averaged transition probability
						// so p is effectively the belief's quantification of T(s', s, a)
						p += getTransitionProbability(s, action, sNext) * b.getBelief(s);
					}
					
					prob += getObservationProbability(action, sNext, obs) * p;
				}
				
				aoProbs[action][obs] = prob;
			}
		}
		
		b.setActionObservationProbabilities(aoProbs);
	}
	
	public BeliefPoint getInitialBelief() {
		return b0;
	}
	
	public void setInitialBelief(BeliefPoint b)
	{
		b0=b;
	}	
	
	public int nextState(int currentState, int action) {
		// Use the active connector instance instead of creating a new one
		// The active instance has noiseInjector properly configured
		iot.DeltaIOTConnector dataConnector = iot.DeltaIOTConnector.activeInstance;
		if (dataConnector == null) {
			log.warn("activeInstance is null in POMDP.nextState(), creating fallback instance");
			dataConnector = new iot.DeltaIOTConnector();
		}
		
		if(action == 0) {
			System.out.println("DTP");
			dataConnector.performDTP(); // decrease transmission power			
		}
		else if(action==1) {
			log.trace("Action: ITP");
			dataConnector.performITP();	 // increase transmission power		
		}
		
		// Note: This method is called during performAction() to determine the next state
		// The action (DTP/ITP) has just been executed, changing the network configuration.
		// At this point in execution:
		// - The previous mote's post-action run has been completed and timestepiot points to it
		// - This previous mote's post-action state serves as the baseline for the current mote
		// - The current mote's action has just been executed but not yet simulated
		// - We use the baseline QoS (previous mote's post-action state) to estimate the next state
		// 
		// Since we're in a sequential loop over motes, each mote's baseline is the previous
		// mote's post-action state. Therefore, timestepiot directly points to the baseline run.
		// 
		// Edge case: For the first mote in timestep 0, timestepiot = 0, so currentRun = 1.
		// Since run 1 doesn't exist yet, waitForQoSDataReady() will return null/empty,
		// and we'll return currentState as fallback (which is correct for the initial state).
		int requestedRun = iot.DeltaIOTConnector.timestepiot;
		
		// Use timestepiot directly to get the previous mote's post-action run, which serves
		// as the baseline for the current mote. This run has complete QoS data for all motes.
		// Math.max(1, ...) ensures we never request run 0 (simulator uses 1-indexed runs).
		int currentRun = Math.max(1, requestedRun);

		// Wait for QoS data to be ready before accessing it to prevent warnings
		ArrayList<QoS> result = iot.QoSDataHelper.waitForQoSDataReady(currentRun, 20, 200);
		if (result == null || result.isEmpty()) {
			log.warn("No QoS data for run {} (requested {}) in nextState(), using current state", currentRun, requestedRun);
			// Return current state as fallback
			return currentState;
		}
		double packetLoss = result.get(result.size()-1).getPacketLoss();
		// This is being performed inside of loop of the motes, so use timestepiot to get QoS for that specific mote
		double energyConsumption = result.get(result.size()-1).getEnergyConsumption();
		
		if(energyConsumption < mecThreshold && packetLoss < rplThreshold) {
			return 0;
		}
		else if(energyConsumption < mecThreshold && packetLoss >= rplThreshold) {
			return 1;
		}
		else if(energyConsumption >= mecThreshold && packetLoss < rplThreshold) {
			return 2;
		}
		else if(energyConsumption >= mecThreshold && packetLoss >= rplThreshold) {
			return 3;
		}
		
		return 0;
	}
	
	///Set it to currentState at the beginning. Each integer indicates the state
	public int getInitialState() {
		// At the start of timestep t (t>0), we need the network state from the END of the previous timestep.
		// That is the last run of timestep t-1, i.e. run = timestepiot (which equals t * numMotes at timestep start).
		// Using run 1 would use stale QoS from the very first run and prevents the belief from reflecting reality.
		int runForTimestepStart = Math.max(1, iot.DeltaIOTConnector.timestepiot);
		ArrayList<QoS> result = iot.QoSDataHelper.waitForQoSDataReady(runForTimestepStart, 20, 100);
		
		if (result == null || result.isEmpty()) {
			System.err.println("Warning: No QoS data available for run " + runForTimestepStart + " in getInitialState(), using default state 0");
			return 0;
		}
		
		log.trace("getInitialState result size: {}", result.size());
		// Get PL and EC at current timestep
		double packetLoss = result.get(result.size()-1).getPacketLoss();
		double energyConsumption = result.get(result.size()-1).getEnergyConsumption();
		
		if(energyConsumption < mecThreshold && packetLoss < rplThreshold) {
			return 0;
		}
		else if(energyConsumption < mecThreshold && packetLoss >= rplThreshold) {
			return 1;
		}
		else if(energyConsumption >= mecThreshold && packetLoss < rplThreshold) {
			return 2;
		}
		else if(energyConsumption >= mecThreshold && packetLoss >= rplThreshold) {
			return 3;
		}
		
		return 0;
	}
	
	public int getCurrentState() {
		return currentState;
	}
	
	public void setCurrentState(int s) {
		currentState=s;
	}
	
	/** Set MEC (energy consumption) threshold for state discretisation; satisfied when energy < threshold. */
	public void setMecThreshold(double mecThreshold) {
		this.mecThreshold = mecThreshold;
	}
	
	public double getMecThreshold() {
		return mecThreshold;
	}
	
	/** Set RPL (packet loss ratio) threshold for state discretisation; satisfied when packetLoss < threshold. */
	public void setRplThreshold(double rplThreshold) {
		this.rplThreshold = rplThreshold;
	}
	
	public double getRplThreshold() {
		return rplThreshold;
	}
	
	
	public int getObservation(Integer action, Integer statePrime) {
		// TODO Auto-generated method stub
		
		log.trace("getObservation: mote {}", iot.DeltaIOTConnector.selectedmote.getMoteid());
		for (Link link : iot.DeltaIOTConnector.selectedmote.getLinks()) {	
			log.trace("Link SNR={}, distribution={}", link.getSNR(), link.getDistribution());
		//if (link.getSNR() > 0 && link.getPower()>0) {
			if (link.getSNR() > 0) {
				//DeltaIOTConnector.selectedmote=m;
				//DeltaIOTConnector.selectedlink=link;
				return 2;
			}
			else if (link.getSNR() == 0) {
				//DeltaIOTConnector.selectedmote=m;
				//DeltaIOTConnector.selectedlink=link;
				return 1;
			}
			//else if (link.getSNR()<0 && link.getPower()<15) {
			else if (link.getSNR() <0) {
				//DeltaIOTConnector.selectedmote=m;
				//DeltaIOTConnector.selectedlink=link;
				return 0;
			}
		}
		return 0;
	}			
}
	
	

	