import rdm.management.Probe;

import java.util.concurrent.ThreadLocalRandom;

import rdm.management.Effector;
import rdm.management.NetworkManagment;
import rdm.network.Monitorables;
import rdm.network.Topology;
import rdm.network.TopologyList;

public class MAPE_KLoop {
	
	Probe probe;
	Effector effector;
	
	public MAPE_KLoop(Probe Probe, Effector effector)
	{
		this.probe=Probe;
		this.effector=effector;
		
	}
	//Monitor the network using probe functions
	public void monitor(int simulation_timestep)
	{
		//get monitorables method executes a single run and return the monitorables for the current simulation run
		Monitorables m=probe.getMonitorables();
		analysisAndPlanning(simulation_timestep, m);
	}
	
	//Analysis and planning for the adaptation
	public void analysisAndPlanning(int simulation_timestep,Monitorables m)
	{
		String selected_topology;
		
		if (probe.getBandwidthConsumption()>m.getThresholdBandwidthConsumption()||probe.getTimeToWrite()>m.getThresholdTimeToWrite())
		{
			
			int random_topology=(int)ThreadLocalRandom.current().nextDouble(0,TopologyList.topologies.size());
			selected_topology=TopologyList.topologies.get(random_topology).getTopologyName();
			
			execute(simulation_timestep,selected_topology);
			
			
			
		}
		else if(probe.getActiveLinks()>m.getThresholdActiveLinks())
		{
			
			int random_topology=(int)ThreadLocalRandom.current().nextDouble(0,TopologyList.topologies.size());
			selected_topology=TopologyList.topologies.get(random_topology).getTopologyName();
			execute(simulation_timestep,selected_topology);
						
			
		}
		else
		{
			int random_topology=(int)ThreadLocalRandom.current().nextDouble(0,TopologyList.topologies.size());
			selected_topology=TopologyList.topologies.get(random_topology).getTopologyName();
			
						
			execute(simulation_timestep,selected_topology);
						
			
			
		}
	
		
	}
	
	//Execute the adapatation using functions of the effector
	public void execute(int simulation_timestep,String selected_topology)
	{
		effector.setNetworkTopology(simulation_timestep,selected_topology);
		
	}
	
	
	public void run(int simulation_timestep)
	{
		monitor(simulation_timestep);
	}
}
