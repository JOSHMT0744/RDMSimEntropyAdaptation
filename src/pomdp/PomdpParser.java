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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import solver.BeliefPoint;

public class PomdpParser {
	/**
	 * Parse a .POMDP file and create POMDP object
	 * @param filename full path to the .POMDP file
	 * @return a POMDP object
	 */
	
	private static Map<String, Integer> mapToInts(List<String> objects) {
		Map<String, Integer> map = new HashMap<>();
		for(int i = 0; i < objects.size(); i++) {
			map.put(objects.get(i).toUpperCase(), i);
		}
		return map;	
	}
	
	
	public static POMDP readPOMDP(String filename) throws IOException {
		/*
		 * filename, nStates, nActions, nObservations, 
		 * discountFactor, rewardFunction, transitionFunction
		 * observationFunction, actionLabels, b0
		 */
		System.out.println();
		System.out.println("=== READ POMDP FILE ===");
		System.out.println("File: "+filename);
		
		List<String> lines = Files.readAllLines(Paths.get(filename));
		
		Map<String, Integer> actionMap = new HashMap<>();
		Map<String, Integer> stateMap = new HashMap<>();
		Map<String, Integer> obsMap = new HashMap<>();
		HashMap<Integer, String> actionLabels = new HashMap<>();
		
		int numStates = 0;
		int numActions = 0;
		int numObservations = 0;
		
		double discount = 0;
		
		List<String> states = new ArrayList<>();

		int lineIndex = -1;
		
		while (lineIndex < lines.size()-1) {
			lineIndex ++;
			String line = lines.get(lineIndex).trim();
			line = line.toLowerCase();
			
			if (line.isEmpty() || line.startsWith("#")) {
				continue;
			}
			
			// Discount
			if (line.startsWith("discount:")) {
				System.out.println("discount");
				discount = Double.parseDouble(line.split(":")[1].trim());
				continue;
			}
			
			// number of states
			if (line.startsWith("states:")) {
				states = Arrays.asList(line.split(":")[1].trim().split("\\s+"));
				stateMap = mapToInts(states);
				numStates = states.size();
				continue;
			} 
			
			//number of Actions
			if (line.startsWith("actions:")) {
				List<String> actions = Arrays.asList(line.split(":")[1].trim().split("\\s+"));
				System.out.println("Actions: " + actions);
				actionMap = mapToInts(actions);
				numActions = actions.size();
				
				for (int i = 0; i < numActions; i++) {
					actionLabels.put(actionMap.get(actions.get(i)),actions.get(i));
				}
			}
			
			// number of observations
			if (line.startsWith("observations:")) {
				List<String> observations = Arrays.asList(line.split(":")[1].trim().split("\\s+"));
				System.out.println("Observations list: " + observations);
				obsMap = mapToInts(observations);
				numObservations = observations.size();
				}
			}
		
		System.out.println("here\n");
		double[] beliefEntries = new double[numStates];
		double[][][] transitionFunction = new double[numStates][numActions][numStates];
		double[][][] observationFunction = new double[numActions][numStates][numObservations];
		double[][] rewardFunction = new double[numStates][numActions];
		
		// Resuming file parsing
		lineIndex = 0;
		while (lineIndex < lines.size()-1) {
			lineIndex ++;
			String line = lines.get(lineIndex).trim();
			//line = line.toLowerCase();
			
			if (line.isEmpty() || line.startsWith("#")) {
				continue;
			}
			
			// initial belief of impact of adaptation actions
			if (line.startsWith("Start:") || line.startsWith("start:")) {				
				beliefEntries = Arrays.stream(line.split(":")[1].trim().split("\\s+"))
						.mapToDouble(Double::parseDouble)
						.toArray();
				}

			String curActionStr;
			int curAction;
			double prob = 0;
			
			// Create transition function
			if (line.startsWith("T:")) {
				String[] transTok = line.split(":");
				
				curActionStr = transTok[1].trim();
				curAction = actionMap.get(curActionStr);
				
				String fromStateStr = transTok[2].trim();
				int fromState = stateMap.get(fromStateStr);
				
				String toStateStr = transTok[3].trim();
				int toState = stateMap.get(toStateStr);
				
				prob = Double.parseDouble(lines.get(lineIndex + 1).trim());
				
				transitionFunction[fromState][curAction][toState] = prob;
				continue;
			}
				
			if (line.startsWith("O:")) {
				String[] obsTok = line.split(":");
				
				curActionStr = obsTok[1].trim();
				curAction = actionMap.get(curActionStr);
				
				String curStateStr = obsTok[2].trim();
				
				String curObsStr = obsTok[3].trim();
				System.out.println("curObsStr " + curObsStr);
				int obs = obsMap.get(curObsStr);
				System.out.println("obs " + obs);
				
				if (curStateStr.equals("*")) {
					prob = Double.parseDouble(lines.get(lineIndex + 1).trim());

					// Apply this obs prob to all states
					for (int state = 0 ; state < numStates; state++) {
						observationFunction[curAction][state][obs] = prob;
					}
					continue;
					}
				}
			
			// Reward probability table
			if (line.startsWith("R:")) {
				String[] rewTok = line.split(":");
				
				curActionStr = rewTok[1].trim();
				curAction = actionMap.get(curActionStr);
				
				String curStateStr = rewTok[2].trim();
				int curState = stateMap.get(curStateStr);
				
				prob = Double.parseDouble(lines.get(lineIndex + 1).trim());
				System.out.println("prob: " + prob);
				System.out.println("curState: " + curState + " curAction: " + curAction);
				rewardFunction[curState][curAction] = prob;
				continue;
			}
		}
			
		BeliefPoint b0 = new BeliefPoint(beliefEntries);
		System.out.println("numStates " + numStates);
		System.out.println("numActions " + numActions);
		System.out.println("numObservations " + numObservations);
		System.out.println("discount " + discount);
		
		System.out.println("rewardFunction " + Arrays.deepToString(rewardFunction));
		System.out.println("transitionFunction " + Arrays.deepToString(transitionFunction));
		System.out.println("observation function " + Arrays.deepToString(observationFunction));
		System.out.println("initial belief " + Arrays.toString(b0.getBelief()));
		
		// Initialising beliefs		
		double[][][] transitionBeliefCurr = new double[numStates][numActions][numStates];
		double[][][] transitionBeliefReset = new double[numStates][numActions][numStates];
		double[][][] observationBelief = new double[numActions][numStates][numObservations];
		assert transitionFunction != null;
		
		for (int stateIndex = 0; stateIndex < numStates; stateIndex++) {
			for (int actionIndex = 0; actionIndex < numActions; actionIndex++) {
				System.arraycopy(transitionFunction[stateIndex][actionIndex], 0, transitionBeliefCurr[stateIndex][actionIndex], 0, numStates);
				Arrays.fill(transitionBeliefCurr[stateIndex][actionIndex], 1.0);
				Arrays.fill(transitionBeliefReset[stateIndex][actionIndex], 1.0);
			}
		}
		
		for (int actionIndex = 0; actionIndex < numActions; actionIndex++) {
			for (int stateIndex = 0; stateIndex < numStates; stateIndex++) {
				// Enforcing full BA-POMDP by turning observationFunction into a Dirichlet too
				Arrays.fill(observationBelief[actionIndex][stateIndex], 1.0);
			}
		}

		 // As each value of transition belief is initially a probability, adjust this to be a minimum of 1, and scale all other values accordingly
		//  for (int stateIndex = 0; stateIndex < numStates; stateIndex++) {
		// 		for (int actionIndex = 0; actionIndex < numActions; actionIndex++) {
		// 			double alphaMin = Arrays.stream(transitionBeliefCurr[stateIndex][actionIndex]).min().getAsDouble();
		// 			for (int nextStateIndex = 0; nextStateIndex < numStates; nextStateIndex++) {
		// 				transitionBeliefCurr[stateIndex][actionIndex][nextStateIndex] = transitionBeliefCurr[stateIndex][actionIndex][nextStateIndex] * Double.valueOf(2*numStates);
		// 			}
		// 		}
		// 	}

		 
		 System.out.println("Starting transition curr belief: ");
		 System.out.println(Arrays.deepToString(transitionBeliefCurr));
		 System.out.println("Starting transition rest belief: ");
		 System.out.println(Arrays.deepToString(transitionBeliefReset));
				
		return new POMDP(filename, 
				numStates, 
				numActions, 
				numObservations, 
				discount, 
				rewardFunction, 
				transitionFunction, 
				observationFunction, 
				actionLabels, 
				b0,
				transitionBeliefReset,
				transitionBeliefCurr,
				observationBelief		
				);				
		}
}

