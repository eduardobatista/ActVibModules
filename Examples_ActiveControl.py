# %%
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from CantileverBeam import CantileverBeam
from Filters import FIR
from Adaptive import FIRNLMS, FIRFxNLMS


# %%
fs = 416.0 # Sampling frequency in Hertz

# Beam characteristics:
npoints = 100 # Number of points in the beam (finite element method)
beamlength = 0.58 # Length of the beam in meters
beamwidth = 0.05 # Width of the beam in meters
beamthickness = 0.006 # Thickness of the beam in meters
dampingfactors = [0.01, 0.01, 0.01, 0.01, 0.01] 


# Positions of sensors and forces:
perturbpos = 30 # Position of the perturbation force, which causes beam vibration.
referencepos = 75 # Position of the acceleration measurement at the beam.
controlpos = 60 # Position of the control force
errorpos = 95 # Position of the error acceleration measurement in the beam

firmem = 1000 # Number of samples for the secondary and feedback paths


# %% Creating Beam instance with 100 points:
cbeam = CantileverBeam(npoints=npoints, width=beamwidth, thickness=beamthickness, 
                        length=beamlength, Tsampling=1.0/fs,
                        damp=dampingfactors)
cbeam.reset()
print("Natural frequencies are:\n",
      ",\n".join(cbeam.freqsHz.astype(str).tolist()),
      " (all in Hz).")

xcoords = np.linspace(0.0, cbeam.length, npoints)
xcoords = np.concatenate((xcoords, xcoords[::-1]))
ycoords = np.array([0.0]*npoints + [beamthickness]*npoints)
fig = go.Figure()
fig.add_trace(go.Scatter(x=xcoords, y=ycoords, fill='toself', mode='lines'))
fig.add_annotation(x=perturbpos*beamlength/npoints, y=beamthickness*1.1, 
            ax=0, ay=-30, text="Perturbation",
            showarrow=True, arrowhead=1)
fig.add_annotation(x=referencepos*beamlength/npoints, y=beamthickness*1.1, 
            ax=0, ay=-50, text="Accel. Measurement",
            showarrow=True, arrowhead=1, arrowside="start")
fig.add_annotation(x=controlpos*beamlength/npoints, y=0, 
            ax=0, ay=30, text="Control Force",
            showarrow=True, arrowhead=1)
fig.add_annotation(x=errorpos*beamlength/npoints, y=0, 
            ax=0, ay=50, text="Error Accel.",
            showarrow=True, arrowhead=1, arrowside="start")
fig.update_layout(title="Cantilever Beam", xaxis_title="x (m)", yaxis_title="y (m)")
fig.update_layout(xaxis=dict(range=[0, beamlength*1.1]), yaxis=dict(range=[-(beamthickness + 0.1), beamthickness + 0.1]))
fig.update_layout(width=600, height=350)
fig.show()



# %% Simulating the beam response to a sinusoidal force:

maxtime = 60.0
vibstart = 10.0 # Start time of the vibration
nsteps = int(maxtime * fs) # Total number of steps
vibfreq = 10.0 # Hertz
th = np.linspace(0.0,maxtime,nsteps) # Time vector
xh = 0.3*np.sin(2*np.pi*th*vibfreq) # Sinusoidal force vector
xh[0:int(fs*vibstart)] = 0.0 # Force set to zero for the first 10 seconds

cbeam.reset()
err = np.zeros(nsteps) # Vibration response
# Running simulation:
for k in range(nsteps):
  err[k] = cbeam.getaccelms2(referencepos)  
  cbeam.setforce(perturbpos,xh[k]) 
  cbeam.update() # Updata for 1 sampling period.

# Plotting the results:
fig = px.line()
fig.add_scatter(x=th, y=xh, name="Força (N)", mode="lines")
fig.add_scatter(x=th, y=err, name="Aceleração (m/s²)", mode="lines")
fig.show()



# %% Active control requires modeling both the secondary and feedback paths:
# The secondary path is the path from the control force to the error sensor.
# The feedback path is the path from the control force to the reference sensor.
# Modeling carried out using the FIRNLMS algorithm.

maxtime = 100.0
nsteps = int(maxtime * fs)

firnlms = FIRNLMS(memorysize=firmem,stepsize=0.15,regularization=1e-3) # Create the FIRNLMS object

# Secondary path via impulse response (ideal but not practical):
wsecimpulse = np.zeros(firmem) # Impulse response vector
cbeam.reset()
cbeam.setforce(controlpos,1.0) # Force is applied at the control position
cbeam.update()
wsecimpulse[0] = cbeam.getaccelms2(errorpos) # Read the acceleration at the error position
cbeam.setforce(controlpos,0.0) # Force is removed
for k in range(1,firmem):
  cbeam.update() # Update the beam for 1 sampling period.
  wsecimpulse[k] = cbeam.getaccelms2(errorpos) # Read the acceleration at the error position


# Secondary path via adaptive modeling (the practical way):
cbeam.reset()
xrandom = np.random.randn(nsteps) # Random force vector
yerror = np.zeros(nsteps) # Error signal vector
for k in range(nsteps):
  cbeam.setforce(controlpos,xrandom[k]) # Force is applied at the control position
  cbeam.update() # Update the beam for 1 sampling period.
  yerror[k] = cbeam.getaccelms2(errorpos) # Read the acceleration at the error position

firnlms.run(insignal=xrandom,outsignal=yerror,maxiter=nsteps) # Run the FIRNLMS algorithm
wsecadaptive = firnlms.ww # Adaptive model of the secondary path

# Comparing the two methods:
fig = px.line()
fig.add_scatter(y=wsecimpulse, name="Impulse response", mode="lines")
fig.add_scatter(y=wsecadaptive, name="Adaptive model", mode="lines")
fig.update_layout(title="Secondary path response (FIR)")
fig.show() # Plot the secondary path coefficients

# %% Feedback path via impulse response (ideal but not practical):

wfbkimpulse = np.zeros(firmem) # Impulse response vector
cbeam.reset()
cbeam.setforce(controlpos,1.0) # Force is applied at the control position
cbeam.update()
wfbkimpulse[0] = cbeam.getaccelms2(referencepos) # Read the acceleration at the error position
cbeam.setforce(controlpos,0.0) # Force is removed
for k in range(1,firmem):
  cbeam.update() # Update the beam for 1 sampling period.
  wfbkimpulse[k] = cbeam.getaccelms2(referencepos) # Read the acceleration at the error position


# Secondary path via adaptive modeling (the practical way):
cbeam.reset()
xrandom = np.random.randn(nsteps) # Random force vector
yerror = np.zeros(nsteps) # Error signal vector
for k in range(nsteps):
  cbeam.setforce(controlpos,xrandom[k]) # Force is applied at the control position
  cbeam.update() # Update the beam for 1 sampling period.
  yerror[k] = cbeam.getaccelms2(referencepos) # Read the acceleration at the error position

firnlms.run(insignal=xrandom,outsignal=yerror,maxiter=nsteps) # Run the FIRNLMS algorithm
wfbkadaptive = firnlms.ww # Adaptive model of the secondary path

# Comparing the two methods:
fig = px.line()
fig.add_scatter(y=wfbkimpulse, name="Impulse response", mode="lines")
fig.add_scatter(y=wfbkadaptive, name="Adaptive model", mode="lines")
fig.update_layout(title="Feedback path response")
fig.show() # Plot the secondary path coefficients



# %% Now, after obtaining the secondary and feedback responses,
# the active control using the FIRFxNLMS algorithm can be performed:

maxtime = 120.0
nsteps = int(maxtime * fs) # Total number of steps
vibstart = 0.0 # Start time of the vibration
controlstart = 30.0 # Start time of the control

controller = FIRFxNLMS(mem=300, memsec=1000) # Create the controller
controller.setSecondary(wsecimpulse) # Set the secondary path
controller.setAlgorithm('NLMS') # Set the algorithm to NLMS
controller.mu = 0.001 # Set the step size
controller.psi = 1e-3 # Set the regularization parameter
controller.reset() # Reset the controller

feedbackfilter = FIR(wfbkimpulse) # Create the feedback filter
feedbackfilter.reset() # Reset the filter

vibfreq = 12.0 # Hertz
th = np.linspace(0.0,maxtime,nsteps) # Time vector
xh = 0.3*np.sin(2*np.pi*th*vibfreq) # Sinusoidal force vector
xh[0:int(fs*vibstart)] = 0.0 # Force is zero for the first 10 seconds

cbeam.reset()
err = np.zeros(nsteps) # Vibration response
yfbk = np.zeros(nsteps) # Vibration response

# Running the simulation:
for k in range(nsteps):
  cbeam.setforce(perturbpos,xh[k]) # force is applied
  cbeam.setforce(controlpos,-controller.y) # Control force is applied

  if th[k] >= controlstart: # Control starts at 30 seconds
    controller.update(cbeam.getaccelms2(errorpos)) 
  yfbk[k] = feedbackfilter.filterstep(-controller.y) # Get the feedback force
  controller.evalout(cbeam.getaccelms2(referencepos) - yfbk[k])

  err[k] = cbeam.getaccelms2(errorpos) # Error acceleration is read

  cbeam.update() # beam is updated

# Plotting the results:
fig = px.line()
fig.add_scatter(x=th, y=xh, name="Perturbation force (N)", mode="lines")
fig.add_scatter(x=th, y=err, name="Beam accelaration (m/s²)", mode="lines")
fig.show()

# %%
