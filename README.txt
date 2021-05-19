2D Method

Currently this code is only set up to run a two dimensional procedure.

The main function for the 2D adaptive method is located at DTQAdaptive.py. 
The inputs for the procedure:
 - NumSteps: The number of time steps you wish to run.
 - minDistanceBetweenPoints (lambda in paper): Used to enforce minimum distances between points
                                               Smaller diffusions require smaller values.
 - maxDistanceBetweenPoints (Lambda in paper): Used to enforce maximum distances between points 
                                               Smaller diffusions require smaller values.
 - h: The time step size
 - degree (beta in paper): Used to adjust the accuracy of the method by maintianing boundary points.
                           A larger value means more accuracy but greater cost.
 - meshRadius: Used to determine the radius of the inital radial mesh
 - drift: The drift function
 - diff: The diffusion function
 - PrintStuff: Default is true. When true we print runtime information, when false, minimal information is printed.

Sample drift and diffustion functions for the methods are set in the file DriftDiffFunctionBank.py
and they can be imported for use. You can also write your own functions. 

Please see the files which start with EXAMPLE_ the see how to run the procedure.
    
    
    
    

 



