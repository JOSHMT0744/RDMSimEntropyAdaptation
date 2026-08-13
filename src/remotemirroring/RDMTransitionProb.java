package remotemirroring;

/**
 * RDMTransitionProb
 */
public class RDMTransitionProb {
	

	
	
	
	public static double MC_MST[]= {0.9,0.88,0.92,0.9,0.85,0.83,0.87,0.85};
	public static double MR_MST[]= {0.91,0.93,0.89,0.91,0.93,0.95,0.91,0.93};
	public static double MP_MST[]= {0.9,0.85,0.92,0.87,0.88,0.83,0.9,0.85};
	
	public static double MC_RT[]= {0.86,0.84,0.88,0.86,0.73,0.71,0.75,0.73};
	public static double MR_RT[]= {0.95,0.97,0.93,0.95,0.97,0.99,0.95,0.97};
	public static double MP_RT[]= {0.82,0.75,0.84,0.77,0.8,0.73,0.82,0.75};
	
	
	
	public static double vec[];
	public static int random_int;
	public static int random_int1;
	public static int random_int2;
	public static int deviation;
	public static int deviation_timesteps;
	
	public RDMTransitionProb()
	{
		vec=new double[8];
		
		
	}
	
	/**
	 * Method to get Transition probabilities for stable scenario
	 * @param casenum
	 * @return
	 */
	public static double[] getCaseRT(int casenum)
	{
		
		vec[0]=(MC_RT[casenum])*(MR_RT[casenum])*(MP_RT[casenum]);
		vec[1]=(MC_RT[casenum])*(MR_RT[casenum])*((1-MP_RT[casenum]));
		vec[2]=(MC_RT[casenum])*((1-MR_RT[casenum]))*(MP_RT[casenum]);
		vec[3]=(MC_RT[casenum])*((1-MR_RT[casenum]))*((1-MP_RT[casenum]));
		
		vec[4]=((1-MC_RT[casenum]))*(MR_RT[casenum])*(MP_RT[casenum]);
		vec[5]=((1-MC_RT[casenum]))*(MR_RT[casenum])*((1-MP_RT[casenum]));
		vec[6]=((1-MC_RT[casenum]))*((1-MR_RT[casenum]))*(MP_RT[casenum]);
		vec[7]=((1-MC_RT[casenum]))*((1-MR_RT[casenum]))*((1-MP_RT[casenum]));
		
		return vec;
		
	}
	
	///stable scenario
	public static double[] getCaseMST(int casenum)
	{
		vec[0]=MC_MST[casenum]*MR_MST[casenum]*MP_MST[casenum];
		vec[1]=MC_MST[casenum]*MR_MST[casenum]*(1-MP_MST[casenum]);
		vec[2]=MC_MST[casenum]*(1-MR_MST[casenum])*MP_MST[casenum];
		vec[3]=MC_MST[casenum]*(1-MR_MST[casenum])*(1-MP_MST[casenum]);
		
		vec[4]=(1-MC_MST[casenum])*MR_MST[casenum]*MP_MST[casenum];
		vec[5]=(1-MC_MST[casenum])*MR_MST[casenum]*(1-MP_MST[casenum]);
		vec[6]=(1-MC_MST[casenum])*(1-MR_MST[casenum])*MP_MST[casenum];
		vec[7]=(1-MC_MST[casenum])*(1-MR_MST[casenum])*(1-MP_MST[casenum]);
		
	/*	System.out.println("Transistion probability");
		for(int i=0;i<vec.length;i++)
		{
			System.out.print(vec[i]+"  ");
		}
		System.out.println();*/
		
		return vec;
		
	}
	
	/**
	 * Method to get Transition probabilities for deterministic scenario
	 * @param casenum
	 * @return
	 */
	public static double[] getCaseMSTDet(int casenum)
	{
		/////change it to redundant topology
		//System.out.println("deviation:::::::::::::::::"+ RDMTransitionProb.random_int+" Time::::"+RDMTransitionProb.deviation_timesteps);
		 double deviationvalue=MR_MST[casenum]*((double)random_int/100.0);
			
		//double deviationvalue1=MC_MST[casenum]*((double)random_int1/100.0);
	    //double deviationvalue2=MP_MST[casenum]*((double)random_int2/100.0);
	    
	    //System.out.println("deviation1"+deviationvalue1+"  "+deviationvalue2);
	    
		vec[0]=(MC_MST[casenum])*(MR_MST[casenum]-deviationvalue)*(MP_MST[casenum]);
		vec[1]=(MC_MST[casenum])*(MR_MST[casenum]-deviationvalue)*((1-MP_MST[casenum]));
		vec[2]=(MC_MST[casenum])*((1-MR_MST[casenum])+deviationvalue)*(MP_MST[casenum]);
		vec[3]=(MC_MST[casenum])*((1-MR_MST[casenum])+deviationvalue)*((1-MP_MST[casenum]));
		
		vec[4]=((1-MC_MST[casenum]))*(MR_MST[casenum]-deviationvalue)*(MP_MST[casenum]);
		vec[5]=((1-MC_MST[casenum]))*(MR_MST[casenum]-deviationvalue)*((1-MP_MST[casenum]));
		vec[6]=((1-MC_MST[casenum]))*((1-MR_MST[casenum])+deviationvalue)*(MP_MST[casenum]);
		vec[7]=((1-MC_MST[casenum]))*((1-MR_MST[casenum])+deviationvalue)*((1-MP_MST[casenum]));
		
		return vec;
		
	}
	
	
	/*public static double[] getCaseMSTDet(int casenum)
	{
		
		 
		 double deviationvalue=MR_MST[casenum]*((double)random_int/100.0);
		// System.out.println(deviationvalue+"  "+random_int);
		    
			vec[0]=(MC_MST[casenum])*(MR_MST[casenum]-deviationvalue)*(MP_MST[casenum]);
			vec[1]=(MC_MST[casenum])*(MR_MST[casenum]-deviationvalue)*((1-MP_MST[casenum]));
			vec[2]=(MC_MST[casenum])*((1-MR_MST[casenum])+deviationvalue)*(MP_MST[casenum]);
			vec[3]=(MC_MST[casenum])*((1-MR_MST[casenum])+deviationvalue)*((1-MP_MST[casenum]));
			
			vec[4]=((1-MC_MST[casenum]))*(MR_MST[casenum]-deviationvalue)*(MP_MST[casenum]);
			vec[5]=((1-MC_MST[casenum]))*(MR_MST[casenum]-deviationvalue)*((1-MP_MST[casenum]));
			vec[6]=((1-MC_MST[casenum]))*((1-MR_MST[casenum])+deviationvalue)*(MP_MST[casenum]);
			vec[7]=((1-MC_MST[casenum]))*((1-MR_MST[casenum])+deviationvalue)*((1-MP_MST[casenum]));
			
		
		
		return vec;
		
	}
	*/
	
	
	public static double[] getCaseRTDet(int casenum)
	{
		/////change it to redundant topology
		//System.out.println("deviation:::::::::::::::::"+ RDMTransitionProb.random_int+" Time::::"+RDMTransitionProb.deviation_timesteps);
	    double deviationvalue1=MC_RT[casenum]*((double)random_int1/100.0);
	    double deviationvalue2=MP_RT[casenum]*((double)random_int2/100.0);
	    
	    //System.out.println("deviation1"+deviationvalue1+"  "+deviationvalue2);
	    
		vec[0]=(MC_RT[casenum]-deviationvalue1)*(MR_RT[casenum])*(MP_RT[casenum]-deviationvalue2);
		vec[1]=(MC_RT[casenum]-deviationvalue1)*(MR_RT[casenum])*((1-MP_RT[casenum])+deviationvalue2);
		vec[2]=(MC_RT[casenum]-deviationvalue1)*((1-MR_RT[casenum]))*(MP_RT[casenum]-deviationvalue2);
		vec[3]=(MC_RT[casenum]-deviationvalue1)*((1-MR_RT[casenum]))*((1-MP_RT[casenum])+deviationvalue2);
		
		vec[4]=((1-MC_RT[casenum])+deviationvalue1)*(MR_RT[casenum])*(MP_RT[casenum]-deviationvalue2);
		vec[5]=((1-MC_RT[casenum])+deviationvalue1)*(MR_RT[casenum])*((1-MP_RT[casenum])+deviationvalue2);
		vec[6]=((1-MC_RT[casenum])+deviationvalue1)*((1-MR_RT[casenum]))*(MP_RT[casenum]-deviationvalue2);
		vec[7]=((1-MC_RT[casenum])+deviationvalue1)*((1-MR_RT[casenum]))*((1-MP_RT[casenum])+deviationvalue2);
		
		return vec;
		
	}
	
	public static double[] getCaseRTDet2(int casenum)
	{
		
		
		double deviationvalue1=MC_RT[casenum]*((double)random_int1/100.0);
	    double deviationvalue2=MP_RT[casenum]*((double)random_int2/100.0);
	    
		 double deviationvalue=MR_RT[casenum]*((double)random_int/100.0);
		// System.out.println(deviationvalue+"  "+random_int);
		    
			vec[0]=(MC_RT[casenum]-deviationvalue1)*(MR_RT[casenum]-deviationvalue)*(MP_RT[casenum]-deviationvalue2);
			vec[1]=(MC_RT[casenum]-deviationvalue1)*(MR_RT[casenum]-deviationvalue)*((1-MP_RT[casenum])+deviationvalue2);
			vec[2]=(MC_RT[casenum]-deviationvalue1)*((1-MR_RT[casenum])+deviationvalue)*(MP_RT[casenum]-deviationvalue2);
			vec[3]=(MC_RT[casenum]-deviationvalue1)*((1-MR_RT[casenum])+deviationvalue)*((1-MP_RT[casenum])+deviationvalue2);
			
			vec[4]=((1-MC_RT[casenum])+deviationvalue1)*(MR_RT[casenum]-deviationvalue)*(MP_RT[casenum]-deviationvalue2);
			vec[5]=((1-MC_RT[casenum])+deviationvalue1)*(MR_RT[casenum]-deviationvalue)*((1-MP_RT[casenum])+deviationvalue2);
			vec[6]=((1-MC_RT[casenum])+deviationvalue1)*((1-MR_RT[casenum])+deviationvalue)*(MP_RT[casenum]-deviationvalue2);
			vec[7]=((1-MC_RT[casenum])+deviationvalue1)*((1-MR_RT[casenum])+deviationvalue)*((1-MP_RT[casenum])+deviationvalue2);
			
		
		
		
		
		
		
		
		return vec;
		
	}
	
	public static double[] getCaseMSTDet2(int casenum)
	{
		/////change it to redundant topology
		//System.out.println("deviation:::::::::::::::::"+ RDMTransitionProb.random_int+" Time::::"+RDMTransitionProb.deviation_timesteps);
		 double deviationvalue=MR_MST[casenum]*((double)random_int/100.0);
			
		double deviationvalue1=MC_MST[casenum]*((double)random_int1/100.0);
	    double deviationvalue2=MP_MST[casenum]*((double)random_int2/100.0);
	    
	    //System.out.println("deviation1"+deviationvalue1+"  "+deviationvalue2);
	    
		vec[0]=(MC_MST[casenum]-deviationvalue1)*(MR_MST[casenum]-deviationvalue)*(MP_MST[casenum]-deviationvalue2);
		vec[1]=(MC_MST[casenum]-deviationvalue1)*(MR_MST[casenum]-deviationvalue)*((1-MP_MST[casenum])+deviationvalue2);
		vec[2]=(MC_MST[casenum]-deviationvalue1)*((1-MR_MST[casenum])+deviationvalue)*(MP_MST[casenum]-deviationvalue2);
		vec[3]=(MC_MST[casenum]-deviationvalue1)*((1-MR_MST[casenum])+deviationvalue)*((1-MP_MST[casenum])+deviationvalue2);
		
		vec[4]=((1-MC_MST[casenum])+deviationvalue1)*(MR_MST[casenum]-deviationvalue)*(MP_MST[casenum]-deviationvalue2);
		vec[5]=((1-MC_MST[casenum])+deviationvalue1)*(MR_MST[casenum]-deviationvalue)*((1-MP_MST[casenum])+deviationvalue2);
		vec[6]=((1-MC_MST[casenum])+deviationvalue1)*((1-MR_MST[casenum])+deviationvalue)*(MP_MST[casenum]-deviationvalue2);
		vec[7]=((1-MC_MST[casenum])+deviationvalue1)*((1-MR_MST[casenum])+deviationvalue)*((1-MP_MST[casenum])+deviationvalue2);
		
		return vec;
		
	}
	
	
	public static double[][][] getTransitionFunction()
	{
		double[][][] transitionfunction=new double[8][2][8];
		int totalstates=8;
		int totalactions=2;
		double vecnew[];
		
		for(int cstate=0;cstate<totalstates;cstate++)
		{
			//System.out.println("Current State: S"+cstate);
			for(int action=0;action<totalactions;action++)
			{
				//System.out.println("Action : "+action);
				if(action==0)
				{
					vecnew=getCaseMST(cstate);
					
					
				}
				else
				{
					vecnew=getCaseRT(cstate);
					
				}
				
				for(int stateprime=0;stateprime<totalstates;stateprime++)
				{
					transitionfunction[cstate][action][stateprime]=vecnew[stateprime];
					//System.out.print("S"+stateprime+" ");
					//System.out.print(transitionfunction[cstate][action][stateprime]+" ");
				}
				//System.out.println();
				
			}
		}
	
		return transitionfunction;
	}
	
	
	/*
	* function to get transition function for different cases of the system
	*/
	public static double[][][] getTransitionFunctionCase(int det_case)
	{
		double[][][] transitionfunction=new double[8][2][8];
		int totalstates=8;
		int totalactions=2;
		double vecnew[]=new double[8];
		
		for(int cstate=0;cstate<totalstates;cstate++)
		{
			//System.out.println("\nCurrent State S"+cstate);
			for(int action=0;action<totalactions;action++)
			{
				if(action==0)
				{
					if(det_case==1||det_case==3)
					{
						
						vecnew=getCaseMSTDet(cstate);
					}
					else if(det_case==2||det_case==5)
					{
						vecnew=getCaseMST(cstate);
					}
					else if(det_case==4|| det_case==6)
					{
						vecnew=getCaseMSTDet2(cstate);
					}
					
					
					
				}
				else
				{
					if(det_case==1||det_case==4)
					{
						vecnew=getCaseRT(cstate);
					}
					else if(det_case==2||det_case==3)
					{
						vecnew=getCaseRTDet(cstate);
					}
					else if(det_case==5|| det_case==6)
					{
						vecnew=getCaseRTDet2(cstate);
					}
					
					
				}
				
				//System.out.println("Action : "+action);
				for(int stateprime=0;stateprime<totalstates;stateprime++)
				{
					
					transitionfunction[cstate][action][stateprime]=vecnew[stateprime];
					//System.out.print("S"+stateprime+" ");
					//System.out.print(transitionfunction[cstate][action][stateprime]+" ");
					
				}
				//System.out.println();
				
			}
		}
	
		return transitionfunction;
	}
	
	

	

}
