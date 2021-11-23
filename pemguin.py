import numpy as np
#%matplotlib inline
import pylab as plt
import sys
from scipy.optimize import minimize
import pandas as pd
eps = sys.float_info.epsilon
def norm_based_tf(x, encoder, average, center, width, height, boost=0.0001):
    # center is a normalized set of weights (pointing from avg to the preferred stimulus)
    diffA = x-average
    diffB = np.linalg.norm(diffA) #np.sqrt(np.sum((diffA)**2))
    # modified sigmoid with no NaNs
    #return height * (1/(1+np.exp(-diffB))) * ((np.dot(diffA, center))/(2*diffB + eps) + 0.5)**width + boost
    # original
    return height * diffB * (((np.dot(diffA, center))/(2*diffB) + 0.5)**width) + boost

def norm_based_tf_log(x, encoder, average, center, width, height, boost=0.0001):
    # center is a normalized set of weights (pointing from avg to the preferred stimulus)
    diffA = x-average
    diffB = np.linalg.norm(diffA) #np.sqrt(np.sum((diffA)**2))
    # modified sigmoid with no NaNs
    #return height * (1/(1+np.exp(-diffB))) * ((np.dot(diffA, center))/(2*diffB + eps) + 0.5)**width + boost
    # original
    return height * np.log(diffB) * (((np.dot(diffA, center))/(2*diffB) + 0.5)**width) + boost


def multi_dim_tf(x, encoder, center, width, height, boost=.0001):
    #return np.exp(-np.sum((np.array(x)-center)**2)/(2*width**2))*height + boost
    #return np.e**((np.sum((x-center)/width)**2)/-2.)*height + boost
    return np.e**(np.sum(((np.array(x)-center)/width)**2)/-2.)*height + boost

def tuning_function(x, encoder, center, width, height, boost=.0001):
    #return np.e**((((x-center)/width)**2)/-2.)*height + boost
    return np.exp(((x-center)**2)/(-2*width**2))*height + boost

def circular_tuning_function(x, encoder, ctf, **kwargs):
    # ie, ctf=tuning_function()
    # only works with 1 dimension
    dw = encoder.domain[1] - encoder.domain[0]
    return np.max([ctf(x, encoder, **kwargs), ctf(x-dw, encoder, **kwargs), ctf(x+dw, encoder, **kwargs)])


def poisson(func, encoder, **kwargs):
    def wrap(x, **kwargs):
        if kwargs:
            return np.random.poisson(func(x, **kwargs))
        else:
            return np.random.poisson(func(x))
    return wrap

def random_stimulus(domain):
    if type(domain[0]) is not int:
        stim = np.zeros(len(domain))
        for i,dim in enumerate(domain):
            stim[i] = (np.random.random()*(dim[1]-dim[0]) + dim[0])
        return stim
    else:
        return np.random.random()*(domain[1]-domain[0]) + domain[0]

def gaussian_ext(func, encoder, **kwargs):
    def wrap(x, noise_stimuli=0, noise_strength=0.0, **kwargs):
        stimuli = [x]+[random_stimulus(encoder.domain) for i in range(noise_stimuli)]
        responses = None#np.zeros(len(encoder.channels))
        for i,s in enumerate(stimuli):
            if kwargs:
                val = np.array(func(s, **kwargs),dtype=float).reshape(-1, len(encoder.channels))
            else:
                val = np.array(func(s),dtype=float).reshape(-1, len(encoder.channels))
            if i > 0:
                val*=np.random.normal(noise_strength, noise_strength)/noise_stimuli
            else:
                val*=1-noise_strength
            if responses is None:
                responses = np.zeros(val.shape)
            responses += val
        return responses/len(stimuli)
    return wrap

class Channel(object):
    def __init__(self, encoder, tf, **kwargs):
        self.kwargs = kwargs
        self.tf = tf
        self.encoder = encoder
        
    def respond(self, x, encoder=None, **kwargs):
        if encoder == None:
            encoder = self.encoder
        kw = self.kwargs.copy()
        for k in kwargs:
            kw[k] = kwargs[k]
        return self.tf(x, encoder, **kw)

class Encoder(object):
    def __init__(self, name=None, neural_noise = poisson, external_noise = gaussian_ext, **noise_args):
        self.channels = []
        self.domain = None
        self.domain_width = 0
        self.name = name
        if neural_noise:
            self.sample = neural_noise(self.sample, self, **noise_args)
        if external_noise:
            self.sample = external_noise(self.sample, self, **noise_args)
    def __str__(self):
        if self.name:
            return self.name
        else:
            return object.__str__(self)
    def scale_channel_attributes(self, attr, scale, location=None, falloff_distance=None, center_attr = "center"):
        for ch in self.channels:
            dist = 0
            if location is not None:
                if isinstance(location, float) or isinstance(location, int):
                    dist = abs(location-ch.kwargs[center_attr])
                else:
                    #direction = location-ch.kwargs[center_attr]
                    dist = angle(location, ch.kwargs[center_attr])
            if location is None or falloff_distance is None or dist < falloff_distance:
                distance_scale = 1
                if falloff_distance is not None:
                    distance_scale =1-dist/falloff_distance
                ch.kwargs[attr] *= (1+(scale-1)*distance_scale)
            
    def shift_channels(self, scale, location, falloff_distance=None, normalize=True, center_attr = "center"):
        for ch in self.channels:
            # the movement direction
            direction = location-ch.kwargs[center_attr]
            if isinstance(location, float) or isinstance(location, int):
                dist = abs(location-ch.kwargs[center_attr])
            else:
                dist = angle(location, ch.kwargs[center_attr])
            if dist!=0 and (falloff_distance is None or dist < falloff_distance):
                distance_scale = 0.5
                if falloff_distance is not None:
                    distance_scale =1-dist/falloff_distance
                ch.kwargs[center_attr] += direction*((scale)*(4*(distance_scale*(1-distance_scale))))
                if normalize: # normalize the direction
                    ch.kwargs[center_attr] /= np.linalg.norm(ch.kwargs[center_attr])

    def multi_domain_fill(self, current_domain, remaining_domain, current_center, tuning_function, channel_defaults):
        steps = current_domain[1] - current_domain[0]
        if len(current_domain) >2:
            steps = current_domain[2]
        make_channels = False
        if len(remaining_domain) == 0:
            make_channels = True
        dw = current_domain[1] - current_domain[0]
        for pos in np.arange(current_domain[0], current_domain[1], dw/steps):
            if make_channels:
                self.channels.append(Channel(self, tuning_function, center=np.array(current_center+(pos,)), **channel_defaults))
            else:
                self.multi_domain_fill(remaining_domain[0], remaining_domain[1:], current_center+(pos,), tuning_function, channel_defaults)
    
    def fill_space(self, domain, tuning_function, **channel_defaults):
        # this function just sets center; 
        #  use Channel.static_args for other defaults
        self.channels = []
        self.domain = domain
        if type(domain[0]) is not int:
            # multidim
            self.multi_domain_fill(domain[0], domain[1:], tuple(), tuning_function, channel_defaults)
        else:
            self.domain_width = domain[1]-domain[0]
            count = int(self.domain_width)
            if len(domain) > 2:
                count = domain[2]
            for center in np.arange(domain[0], domain[1], self.domain_width/count):
                self.channels.append(Channel(self, tuning_function, center=center, **channel_defaults))

    def fill_axis(self, domain_start, domain_end, tuning_function, channels=20, **channel_defaults):
        self.channels = []
        start = np.array(domain_start)
        end = np.array(domain_end)
        self.domain = [start,end]
        self.domain_width = np.sqrt(np.sum((end-start)**2))
        count = channels
        step = 1.0/(count-1)
        center = 0.0
        for i in range(count):
            self.channels.append(Channel(self, tuning_function, center=center*(end-start)+start, **channel_defaults))
            center += step

    def channel_attributes(self, attr):
        return [ch.kwargs[attr] for ch in self.channels]
    
    def get_channel_properties(self):
        return [ch.kwargs for ch in self.channels]

    def set_channel_properties(self, list_kwargs):
        for i,ch in enumerate(self.channels):
            ch.kwargs = list_kwargs[i]

    '''def sample(self, x):
        return np.array([ch.respond(x) for ch in self.channels])'''
    def sample(self, x, repetitions=1):
        return self.raw_responses([x], repetitions)

    def raw_responses(self, stimuli, repetitions=1, *args, **kwargs):
        raw_responses = []
        for ch in self.channels:
            raw_responses.append([np.zeros(repetitions) + ch.respond(s, encoder=self, **kwargs) for s in stimuli])
        raw_responses = np.array(raw_responses)
        return np.transpose(raw_responses,axes=(2,1,0))

    def raw_responses_mass(self, sets_of_stimuli, *args, **kwargs):
        all_raw_responses = []
        for ch in self.channels:
            raw_responses = []
            for stimuli in sets_of_stimuli:
                raw_responses.append([ch.respond(s, encoder=self, **kwargs) for s in stimuli])
            all_raw_responses.append(raw_responses)
        return np.array(all_raw_responses).T

    def plot_channels(self, resolution, single_color=None, stimulus_axis=None, **kwargs):
        if stimulus_axis:
            parallel_axis = np.arange(0,1,1/resolution)
            parallel_axis_multi = parallel_axis
            for i in range(len(stimulus_axis[0])-1):
                parallel_axis_multi = np.vstack((parallel_axis_multi,parallel_axis))
            stimuli = parallel_axis_multi.T*(stimulus_axis[1]-stimulus_axis[0])+stimulus_axis[0]
            rs = self.raw_responses(stimuli, 1, **kwargs)[0]
            if single_color is not None:
                plt.plot(parallel_axis, rs, color=single_color)
            else:    
                plt.plot(parallel_axis, rs)
            plt.show()
        else:
            hdw = self.domain_width/2.
            stimuli = np.arange(-hdw, hdw, self.domain_width/resolution)
            rs = self.raw_responses(stimuli, 1, **kwargs)[0]

            if single_color is not None:
                plt.plot(stimuli, rs, color=single_color)
            else:    
                plt.plot(stimuli, rs)
            plt.show()
        return rs

    def plot_channels_stimuli(self, stimuli, single_color=None, **kwargs):
        parallel_axis = np.arange(0,1.0,1.0/len(stimuli))
        rs = self.raw_responses(stimuli, 1, **kwargs)[0]
        if single_color is not None:
            plt.plot(parallel_axis, rs, color=single_color)
        else:    
            plt.plot(parallel_axis, rs)
        plt.show()
        return rs

def estimate_handler(func):
    def wrap(encoder, x, true_target=None, iterations=1000, domain_start = None, domain_end = None):
        mles = func(encoder, x, iterations, domain_start, domain_end)
        good_estimate = np.mean(mles,axis=0)
        compare = good_estimate
        if true_target is not None:
            compare = true_target
        differences = mles-compare
        std_estimate = np.sqrt(np.mean(np.sum(differences**2,axis=1), axis=0))
        return good_estimate, std_estimate
    return wrap


import numdifftools
import scipy

def standard_draw(x, *args, **kwargs):
    return x

def face_draw(x, avg_face, radius, faces, *args, **kwargs):
    #return get_face2(avg_face, radius, faceA, faceB, faceC, faces, x, 1)
    #return get_face_from_angle(avg_face, radius, faceA, faceC, x)
    return get_face_from_angle3(avg_face, radius, faces, x)

def average_estimates(estimates, drawn_targets, *args):
    results = []
    for i,mles in enumerate(estimates):
        # the maximum likelihood estimate
        good_estimate = np.mean(mles,axis=0)
        compare = drawn_targets[i]
        differences = mles-compare
        # the threshold estimate (uncertainty)
        while len(differences.shape) < 2:
            differences = np.array([differences])
        threshold_estimate = np.sqrt(np.mean(np.sum(differences**2,axis=1), axis=0))
        results.append((good_estimate, threshold_estimate))
    return np.array(results, dtype=object)

def fisher_estimates(estimates, drawn_targets, *args):
    results = []
    for i,row in enumerate(estimates):
        inv = np.array([r for r in row])
        threshold = np.sqrt(1/np.average(inv)) # estimate of the threshold (NOT the MLE mean)
        threshold_sd = (np.std(np.sqrt(1/np.array(inv)))) # sd across the many threshold estimates
        results.append((threshold, threshold_sd))
    return np.array(results, dtype=object)

class Decoder(object):
    def __init__(self, estimates_handler = average_estimates, draw_func=standard_draw, *standard_draw_args, **standard_draw_kwargs):
        self.estimates_handler = estimates_handler
        self.estimates_handler_args = []
        self.draw_func = draw_func
        self.draw_args = standard_draw_args
        self.draw_kwargs = standard_draw_kwargs

    def draw_from(self, x):
        # draw stimuli from the space
        return self.draw_func(x, *self.draw_args, **self.draw_kwargs)

    def decode_wrap(self, target):
        return -self.getlogsimul(self.assumed_sampler, self.draw_from(target), *self.assumed_sample_args, r = self.r, **self.sample_kwargs)
    
    def decode_wrap_mm(self, target):
        # the likelihood is a multidimensional Gaussian
        mean = self.assumed_sampler.raw_responses(*self.assumed_sample_args, relevant_stimulus=self.draw_from(target), **self.sample_kwargs).reshape(-1)
        return -scipy.stats.multivariate_normal.pdf(self.r, mean=mean, cov=self.omega)

    def decoding_method(self, target, observed_sampler, observed_sample_kwargs, assumed_sampler, assumed_sample_args):
        # the default decoding method is mcmc
        # get the encoder's actual sample
        samples = np.array(observed_sampler.sample(**self.sample_kwargs, **observed_sample_kwargs))
        if isinstance(observed_sampler, MeasurementModel):
            samples = samples.T
        self.assumed_sampler = assumed_sampler
        self.assumed_sample_args = assumed_sample_args

        estimates = []
        decode_wrap_func = self.decode_wrap
        if isinstance(observed_sampler, MeasurementModel):
            decode_wrap_func = self.decode_wrap_mm

        for sample in samples:
            self.r = sample
            
            if isinstance(observed_sampler, MeasurementModel):
                self.r = self.r.reshape(-1)
            
            # do mcmc
            if self.estimates_handler is average_estimates:
                mini = None
                while True:
                    # test a sample against what the decoder thinks it should be
                    mini = minimize(decode_wrap_func, target, method='BFGS')
                    if mini.success:
                        break
                    #raise(Exception("Not reaching minimum"))
                    kwargs = observed_sample_kwargs.copy()
                    if "trials_per_stimulus" in kwargs:
                        kwargs['trials_per_stimulus'] = 1
                    else:
                        kwargs['repetitions'] = 1
                    self.r = np.array(observed_sampler.sample(**self.sample_kwargs, **kwargs))
                estimates.append(self.draw_from(mini.x))
            elif self.estimates_handler is fisher_estimates: # Fisher Information
                dfunc = numdifftools.Derivative(decode_wrap_func, n=2)
                inv = abs(dfunc(target))#[abs(dfunc(target)) for i in range(iters)]
                estimates.append(inv)
        return estimates

    def decode(self, encoder, targets, iters=1000, unaware_encoder=None, measurement_model=None, noise_variance=None, **sample_kwargs):
        # targets are numeric, and converted by a draw function
        all_estimates = []
        all_targets = []
        self.sample_kwargs = sample_kwargs
    
        for target in targets:
            x = self.draw_from(target)
            all_targets.append(x)
            assumed_sample_args = None
            observed_sample_kwargs = None
            encoder_assumed_by_decoder = encoder
            if unaware_encoder is not None:
                encoder_assumed_by_decoder = unaware_encoder
            assumed_sampler = encoder_assumed_by_decoder
            observed_sampler = encoder
            if measurement_model is not None:
                self.omega = measurement_model.omega(noise_variance)
                assumed_sample_args = [encoder_assumed_by_decoder]
                assumed_sampler = measurement_model
                observed_sample_kwargs = dict(encoder=encoder, relevant_stimulus=x, trials_per_stimulus=iters)
                observed_sampler = measurement_model
            else:
                assumed_sample_args = []
                observed_sample_kwargs = dict(x=x, repetitions=iters)
            
            # make a number of estimates to average into the true estimate
            estimates = self.decoding_method(target, observed_sampler, observed_sample_kwargs, assumed_sampler, assumed_sample_args)
            all_estimates.append(estimates)
        if self.estimates_handler is not None:
            estimates = self.estimates_handler(np.array(all_estimates), all_targets, *self.estimates_handler_args)
        return estimates



    


    def getlogsimul(self, encoder, target, position=None, repetitions=1, r=None, raw=None, domain_start=None, domain_end=None, *args, **kwargs):
        # what is the log likelihood of the target stimulus at "position" (in the stimulus domain)
        if position is None:
            position=target
        if r is None:
            r = np.array(encoder.sample(x=target, repetitions=repetitions, *args, **kwargs))
        if raw is None:
            #raw = encoder.raw_responses([position], repetitions=repetitions).reshape(repetitions, -1)
            raw = encoder.raw_responses([position], repetitions=repetitions)#.reshape(repetitions, -1)
        log_raw = np.log(raw)
        #log_raw = raw*1#np.log(raw)
        #idx = np.where(raw!=0)
        #log_raw[idx] = np.log(raw[idx])

        sum_r = np.sum(raw, axis=2)
        sum_l = np.sum(r * log_raw, axis=2)
        return (sum_l - sum_r)[0]
    


def fill_circle(encoder, avg, radius, faceA, faceB, faceC, channels_per_spoke, 
                        perimeter_channels, tuning_function, jitter=(0, 0.1), **channel_defaults):
    #get_face(avg, radius, faceA, faceB, faceC, angle_proportion, radial_proportion)
    # need to make channels prefer faces on the circle's spokes from 
    #  the average to angle_proportions .25, .5, and .75.
    # need channels to be spread radially as well.
    #self.channels.append(Channel(self, tuning_function, center=center*(end-start)+start, **channel_defaults))
    # for the preferences are dictated by the weights for Giese and Leopold Eq 2.
    # Weights are the direction to the preferred face from the average face 
    #  (thus, weights = face-avg (then normalized, because it is a unit vector)).
    spokes = [.25,.5,.75]
    if channels_per_spoke != 0:
        for spoke in spokes:
            step = 1.0/channels_per_spoke
            for radial in np.arange(step, 1+step/2,step):
                face = get_face(avg, radius, faceA, faceB, faceC, spoke, radial)
                noise = np.random.normal(*jitter, size=face.shape)
                face = face+noise
                if tuning_function is not norm_based_tf:
                    encoder.channels.append(Channel(encoder, tuning_function, center=face, **channel_defaults))
                else:
                    weights = face-avg
                    if np.sum(weights) != 0:
                        weights /= np.linalg.norm(weights)
                    encoder.channels.append(Channel(encoder, tuning_function, center=weights, **channel_defaults))

    step = 360.0/(perimeter_channels)
    for ang in np.arange(0,360,step):
        face = np.array(get_face_from_angle(avg, radius, faceA, faceB, ang), dtype=np.float64)#get_face(avg, radius, faceA, faceB, faceC, angular, radius)
        face += np.random.normal(*jitter, size=face.shape)
        if tuning_function is not norm_based_tf:
            encoder.channels.append(Channel(encoder, tuning_function, center=face, **channel_defaults))
        else:
            weights = face-avg
            weights /= np.linalg.norm(weights)
            encoder.channels.append(Channel(encoder, tuning_function, center=weights, **channel_defaults))


def fill_partial_circle(encoder, avg, radius, faceA, faceB, faceC, channels_per_spoke, 
                        perimeter_channels, tuning_function, jitter=(0, 0.1), **channel_defaults):
    #get_face(avg, radius, faceA, faceB, faceC, angle_proportion, radial_proportion)
    # need to make channels prefer faces on the circle's spokes from 
    #  the average to angle_proportions .25, .5, and .75.
    # need channels to be spread radially as well.
    #self.channels.append(Channel(self, tuning_function, center=center*(end-start)+start, **channel_defaults))
    # for the preferences are dictated by the weights for Giese and Leopold Eq 2.
    # Weights are the direction to the preferred face from the average face 
    #  (thus, weights = face-avg (then normalized, because it is a unit vector)).
    spokes = [.25,.5,.75]
    if channels_per_spoke != 0:
        for spoke in spokes:
            step = 1.0/channels_per_spoke
            for radial in np.arange(step, 1+step/2,step):
                face = get_face(avg, radius, faceA, faceB, faceC, spoke, radial)
                noise = np.random.normal(*jitter, size=face.shape)
                face = face+noise
                if tuning_function is not norm_based_tf:
                    encoder.channels.append(Channel(encoder, tuning_function, center=face, **channel_defaults))
                else:
                    weights = face-avg
                    if np.sum(weights) != 0:
                        weights /= np.linalg.norm(weights)
                    encoder.channels.append(Channel(encoder, tuning_function, center=weights, **channel_defaults))

    step = 1.0/(perimeter_channels-1)
    for angular in np.arange(0,1+step/2,step):
        face = get_face(avg, radius, faceA, faceB, faceC, angular, radius)
        face += np.random.normal(*jitter, size=face.shape)
        if tuning_function is not norm_based_tf:
            encoder.channels.append(Channel(encoder, tuning_function, center=face, **channel_defaults))
        else:
            weights = face-avg
            weights /= np.linalg.norm(weights)
            encoder.channels.append(Channel(encoder, tuning_function, center=weights, **channel_defaults))

def dist(A,B):
    return np.sqrt(np.sum((A-B)**2))

def angle(A,B):
    # get the angle from the arccos of cosine similarity
    Al = np.linalg.norm(A)
    Bl = np.linalg.norm(B)
    
    sim = np.dot(A,B)/(Al*Bl)
    # account for numerical precision problems,
    # maintain [-1, 1] bounds:
    if abs(sim) > 1:
        sim = np.sign(sim)
    return np.arccos(sim)/np.pi*180

def set_faces_to_radius(avg, faceA, faceB):
    d1 = dist(avg, faceA)
    d2 = dist(avg, faceB)

    # make them the same distance from the average
    radius = d2
    if d1 > d2:
        faceA = avg+(faceA-avg)/d1 * d2
    else:
        faceB = avg+(faceB-avg)/d2 * d1
        radius = d1
    return radius, faceA, faceB

def get_face_C(avg, radius, faceA, faceB):
    #if dist(faceA, avg) != dist(avg, faceB):
    #    raise Exception("The faces must be equidistant from the average.")
    faceC = faceB + faceB-faceA
    # the face isn't rotated far enough if you just use the difference vector
    faceC = avg + (faceC-avg)/dist(faceC,avg) * radius

    # brute force the rotation by using vectors that are definitely on the circle
    # and by using properties of the circle:
    # the distance from A to B must match B to C AND avg to C must be the radius
    distanceAB = dist(faceA,faceB)
    distanceBC = dist(faceB, faceC)
    prev = [distanceBC]
    while distanceBC != distanceAB:
        # move it away from faceB
        faceC = faceB + (faceC-faceB)/distanceBC * distanceAB
        # move it to radius
        faceC = avg + (faceC-avg)/dist(faceC,avg) * radius
        distanceBC = dist(faceB, faceC)
        #print(distanceBC)
        if distanceBC in prev:
            #print("Stopped Due to limited floating point precision.")
            break
        prev.append(distanceBC)
    return faceC

def get_face_from_angle(avg, radius, faceA, faceB, desired_angle):
    # desired_angle: uses angles (0,180) noninclusive
    # faceA and faceB NEED to already be on the circle.
    #if dist(faceA, avg) != dist(avg, faceB):
    #    raise Exception("The faces must be equidistant from the average.")
    direction = faceB-faceA
    if desired_angle < 0:
        # switching face A and B (while keeping the starting point) 
        #  and flipping the sign of desired_angle to +
        #  gives the same effect as a negative angle.
        desired_angle *= -1
        direction *= -1
    
    if desired_angle == 180:
        return avg + (avg-faceA)

    if desired_angle > 180:
        direction *= -1
        desired_angle = abs(desired_angle-360)
    faceC = faceA + direction

    # make the face on the edge
    faceC = avg + (faceC-avg)/dist(faceC,avg) * radius

    # brute force the rotation by using vectors that are definitely on the circle
    # and by using properties of the circle:
    # C must be on the radius, and it must be vaguely in the direction specified by A and B.
    # (A and B must not be on exactly opposite sides of average)
    ang = angle(faceA-avg,faceC-avg)
    prev = [ang]
    while ang != desired_angle:#abs(ang-desired_angle)>0.0000000000001:
        # move it towards or away from faceA based on the ratio 
        #  of the desired angle and the last angle
        faceC = faceA + (faceC-faceA)*(desired_angle/ang)
        # set it at the radius
        faceC = avg + (faceC-avg)/dist(faceC,avg) * radius
        # measure the new angle
        ang = angle(faceA-avg,faceC-avg)
        #print(ang)
        if ang in prev:
            #print("Stopped Due to limited floating point precision.")
            break
        prev.append(ang)
    return faceC


# the angle predetermines the distance with trig
def get_face_from_angle2(avg, radius, faceA, faceB, desired_angle):
    #if dist(faceA, avg) != dist(avg, faceB):
    #    raise Exception("The faces must be equidistant from the average.")
    direction = faceB-faceA
    if desired_angle < 0:
        # switching face A and B (while keeping the starting point) 
        #  and flipping the sign of desired_angle to +
        #  gives the same effect as a negative angle.
        direction *= -1
        desired_angle *= -1

    if desired_angle == 180:
        return avg + (avg-faceA)

    if desired_angle > 180:
        direction *= -1
        desired_angle = abs(desired_angle-360)
    faceC = faceA + direction
    # opp/hyp == sin(t)
    # opp == sin(t)*hyp
    # Because of right triangle things,
    #  it needs to be half the angle and twice the distance,
    #  thus /360, which is angle/pi/180/2
    desired_distance = np.sin(desired_angle*np.pi/360)*radius*2
    # the face isn't rotated far enough if you just use the difference vector
    faceC = avg + (faceC-avg)/dist(faceC,avg) * radius

    # brute force the rotation by using vectors that are definitely on the circle
    # and by using properties of the circle:
    # the distance from A to B must match B to C AND avg to C must be the radius
    distanceAC = dist(faceA, faceC)
    prev = [distanceAC]
    while distanceAC != desired_distance and len(prev) < 100:#abs(distanceAC - desired_distance) > 0.00000000000001:
        # move it away from faceB
        faceC = faceA + (faceC-faceA)/distanceAC * desired_distance
        # move it to radius
        faceC = avg + (faceC-avg)/dist(faceC,avg) * radius
        distanceAC = dist(faceA, faceC)
        if distanceAC in prev or distanceAC==0:
            #print("Stopped Due to limited floating point precision.")
            break
        prev.append(distanceAC)
    return faceC

def get_face(avg, radius, faceA, faceB, faceC, angle_proportion, radial_proportion):
    # retrieve the face given the three faces 
    #  and the proportion from A to C
    #  and from the radius to average.
    # .5 angle_proportion is the category boundary.
    if radial_proportion == 0:
        return avg
    face = None
    if angle_proportion == 0:
        face = faceA
    elif angle_proportion == .5:
        face = faceB
    elif angle_proportion == 1:
        face = faceC
    
    
    full_range = angle(faceA-avg,faceC-avg)
    desired_angle = full_range*angle_proportion
    if desired_angle == 180:
        face = avg - (faceA-avg)
    elif desired_angle > 180:
        desired_angle = desired_angle - 360
    elif desired_angle < -180:
        desired_angle = desired_angle + 360
    # get the face along the radius
    if face is None:
        face = get_face_from_angle2(avg, radius, faceA, faceB, desired_angle)
    # adjust the distance to the average
    if radial_proportion != 1:
        face = avg + (face-avg)*radial_proportion
    return face

def get_face2(avg, radius, faceA, faceB, faceC, faces, angle_proportion, radial_proportion):
    # retrieve the face given the three faces 
    #  and the proportion from A to C
    #  and from the radius to average.
    # .5 angle_proportion is the category boundary.
    if radial_proportion == 0:
        return avg
    face = None
    if angle_proportion == 0:
        face = faceA
    elif angle_proportion == .5:
        face = faceB
    elif angle_proportion == 1:
        face = faceC
    
    
    full_range = angle(faceA-avg,faceC-avg)
    desired_angle = full_range*angle_proportion
    # get the face along the radius
    if face is None:
        face = get_face_from_angle3(avg, radius, faces, desired_angle)
    # adjust the distance to the average
    if radial_proportion != 1:
        face = avg + (face-avg)*radial_proportion
    return face

def get_face_from_angle3(avg, radius, faces, desired_angle):
    while desired_angle > 360:
        desired_angle -= 360
    while desired_angle < 0:
        desired_angle += 360
    # faces has 0 - 359 in integers
    if int(desired_angle) == desired_angle:
        return faces[int(desired_angle)]
    faceA = faces[int(desired_angle)]
    faceB = faces[int(np.ceil(desired_angle))%360]
    desired_angle -= int(desired_angle)
    return get_face_from_angle2(avg, radius, faceA, faceB, desired_angle)

class MeasurementModel(object):
    def __init__(self, measurement_channels, encoder_channelN, covariance = None, uninterested_percent = 0, normalize_weights = True, name = "Measurement Model", *args, **kwargs):
        self.name = name
        self.models = {}
        # the number of measurement channels (voxels, electrodes, etc)
        self.measurement_channels = measurement_channels

        # count the number of channels in an encoding model.
        self.neural_channels = encoder_channelN
        # make the list of means for the measurement channels
        self.means = np.zeros((self.measurement_channels,))
        
        # set up the covariance between measurement channels
        self.covariance = covariance
        if covariance is None:
            # make identity if none is provided
            self.covariance = np.zeros((self.measurement_channels, self.measurement_channels))
            for i, r in enumerate(self.covariance):
                self.covariance[i,i] = 1
                
        # make random (positive) weights given the number
        #  of measurement channels times the number of neural channels
        self.weights = np.random.random((self.measurement_channels, self.neural_channels))
        
        if uninterested_percent > 0:
            uninterested_count = self.measurement_channels*uninterested_percent/100.0
            zeroed = np.random.choice(np.arange(self.measurement_channels), replace=False, size=(uninterested_count,))
            self.weights[zeroed, :]=0
            
        # normalize the weights (within each voxel/measurement channel) so they add up to 1 by dividing by the magnitude.
        if normalize_weights:
            for i,row in enumerate(self.weights):
                summed = np.sum(self.weights[i, :])
                if summed > 0:
                    self.weights[i, :] /= summed
    
    def omega(self, noise_variance):
        return self.covariance + noise_variance*np.matmul(self.weights,self.weights.T)

    def make_random_corrs(self):
        rando_data = np.random.normal(scale=1, size=self.covariance.shape)
        return np.corrcoef(rando_data)
    
    def make_random_SDs(self, bounds):
        sds = np.random.random(size=self.covariance.shape[0])*(bounds[1]-bounds[0])+bounds[0]
        return sds
    
    def set_covariance_matrix(self, SDs="random", correlation_matrix="random"):
        if type(correlation_matrix) is str and correlation_matrix == "random":
            correlation_matrix = self.make_random_corrs()
        
        if type(SDs) == str and SDs == "random":
            SDs = self.make_random_SDs((.5,1))

        # set the noise covariance matrix based on the formula:
        #  cov(X,Y) = SDx * SDy *corr(X,Y)
        for i in range(self.covariance.shape[0]):
            for k in range(self.covariance.shape[1]):
                if i == k:
                    self.covariance[i,i] = SDs[i]**2
                else:
                    self.covariance[i,k] = SDs[i]*SDs[k]*correlation_matrix[i,k]
                    self.covariance[k,i] = SDs[i]*SDs[k]*correlation_matrix[i,k]
        
        return self.covariance

    def raw_responses(self, encoder, relevant_stimulus, repetitions=1, *args, **kwargs):
        # get weighted, noiseless voxel responses
        raws = encoder.raw_responses(stimuli=[relevant_stimulus], repetitions=repetitions, **kwargs)
        # store it in the big matrix
        C1 = np.transpose(raws.T, (1,0,2))[0]
        # M x C * C x T ==> M x T
        weighted = np.matmul(self.weights, C1)
        return weighted

    def sample(self, encoder, relevant_stimulus, trials_per_stimulus=1, **kwargs):
        # C1 shape: trials_per_stimulus X neural_channels
        
        # sample a single measurement channel?
        sampled = np.array(encoder.sample(x=relevant_stimulus, repetitions=trials_per_stimulus, **kwargs))

        # store it in the big matrix
        C1 = sampled.T
        # M x C * C x T ==> M x T
        weighted = np.matmul(self.weights, C1)

        # add measurement noise
        # M x T
        rando = np.random.multivariate_normal(self.means, self.covariance, trials_per_stimulus).T
        # M x T + M x T
        return weighted + rando
    
    def __generate_artificial_data__(self, stimuli, trials_per_stimulus, set_count, *encoders, **kwargs):
        B1 = np.zeros((len(encoders), len(stimuli), self.measurement_channels, trials_per_stimulus))
        full_dataset = np.zeros((len(encoders)*len(stimuli)*
                         trials_per_stimulus*set_count,
                         self.measurement_channels+4), dtype=object)
        
        print("Data size", full_dataset.shape)
        total_rows = float(full_dataset.shape[0])
        row_index = 0
        # for each dataset
        for set_i in range(set_count):
            # for each irrelevant level/encoder
            for ri, level in enumerate(encoders):
                if self.neural_channels != len(level.channels):
                    print("WARNING {0} has {1} Channels, but Measurement model wants {2} Channels".format(level, len(level.channels), self.neural_channels))
                print("Starting {0} for dataset {1}".format(level, set_i+1))
                # for each stimulus
                for si,s in enumerate(stimuli):
                    # for each measurement channel
                    for mc in range(self.measurement_channels):
                        # for every trial for the stimulus:
                        #for t in range(trials_per_stimulus):
                        #    B1[ri,si,:, t] = self.sample(level, s, mc)
                        B1[ri,si,:, :] = self.sample(level, s, trials_per_stimulus, **kwargs)

                    full_dataset[row_index:row_index+B1[ri,si,:].T.shape[0], 4:] = B1[ri,si,:].T
                    full_dataset[row_index:row_index+B1[ri,si,:].T.shape[0], :4] = (set_i+1, 0, level, s)
                    row_index += B1[ri,si,:].T.shape[0]
                    print("{0:.4f}".format(row_index/total_rows), end='\r')
        trials = np.array(list(np.arange(1, (full_dataset.shape[0])/set_count+1))*set_count)
        full_dataset[:,1]=trials
        col_labels = ["DatasetNumber", "Trial","Irrelevant", "Relevant"]+["measurement_channel_{0}".format(i) for i in range(self.measurement_channels)]
        
        return full_dataset, col_labels
    
    def generate_data(self, data_file_name = "generated_default_name.csv", stim_count = 8, trials_per_stimulus = 100, set_count=2, *encoders, **kwargs):
        domain = encoders[0].domain
        for e in encoders[1:]:
            d = e.domain
            if d[0] != domain[0] or d[1] != domain[1]:
                print("WARNING: Encoders have different domains: {} vs {}".format(domain, d))
        stimuli = np.arange(domain[0], domain[1], float(domain[1]-domain[0])/stim_count)
        print(stimuli)
        simulated_data, col_labels = self.__generate_artificial_data__(stimuli, trials_per_stimulus, set_count, *encoders, **kwargs)

        df = pd.DataFrame(simulated_data, columns=col_labels)
        df.to_csv(data_file_name)
        return df

from sklearn.svm import NuSVC
from sklearn.model_selection import LeavePOut, KFold
from scipy.stats import binom, binom_test, norm
import pandas as pd
import numpy as np
import pylab as plt
from statsmodels.stats.proportion import proportion_confint, proportions_chisquare, proportions_ztest
from statsmodels.stats.multitest import multipletests

# make the background white so the plots copy easily
#plt.rcParams['figure.facecolor']='white'
def labelfy(arr):
    return np.array(arr, dtype=str)

def get_scores(idx, training_set, training_response_set, nu):
    def labelfy(arr):
        return np.array(arr, dtype=str)
    train_index, test_index = idx
    X_train, X_test = training_set[train_index], training_set[test_index]
    y_train, y_test = training_response_set[train_index], training_response_set[test_index]
    # some nu's aren't feasible, but don't crash over it, just give a score of 0
    try:
        clf = NuSVC(gamma='scale', kernel='linear', nu=nu)
        clf.fit(X_train, labelfy(y_train))
        score = clf.score(X_test, labelfy(y_test))
        return score
    except Exception as e:
        print(e)
        return 0

    

def get_best_nu(data_file, cut_off = 5, split_type=KFold, kshuffle=False):
    df = pd.read_csv(data_file)
    datasetsTraining = {}
    
    training = df.query("DatasetNumber==1")
    
    irrelevant_dims = np.unique(training["Irrelevant"].values)

    for key in irrelevant_dims:
        datasetsTraining["{} training".format(key)] = training.query("Irrelevant=='{}'".format(key))
    
    # find the best nu
    best_score = -1
    best_nu = -1
    splitter = None
    if split_type is LeavePOut:
        splitter = LeavePOut(p=20)
    elif split_type is KFold:
        splitter = KFold(n_splits=5, shuffle=kshuffle)
    for train in datasetsTraining:
        training_set = datasetsTraining[train].values[:, cut_off:]
        print(training_set.shape)
        training_response_set = datasetsTraining[train]["Relevant"].values
        
        for nu in np.arange(0.1, 1, .1):
            idx = [i for i in splitter.split(training_set)]
            scores = [get_scores(i, training_set, training_response_set, nu) for i in idx]#
            mean = np.mean(scores)
            print(train, "Nu:", nu, "score:", mean)
            if best_score < mean:
                best_nu = nu
                best_score = mean
    print("Best Nu", best_nu)
    return best_nu

def setup_decoders(data_file, best_nu, sd_scale, alpha = .05, cut_off = 5):
    df = pd.read_csv(data_file)
    datasetsTraining = {}
    datasetsTesting = {}
    classifiers = {}
    classifier_scores = {}
    
    training = df.query("DatasetNumber==1")
    testing = df.query("DatasetNumber==2")

    
    stimuli = np.sort(np.unique(training["Relevant"].values))
    irrelevant_dims = np.unique(training["Irrelevant"].values)

    for key in irrelevant_dims:
        datasetsTraining["{} training".format(key)] = training.query("Irrelevant=='{}'".format(key))
        datasetsTesting["{} testing".format(key)] = testing.query("Irrelevant=='{}'".format(key))
    
    z = norm.ppf(1-alpha/2)
    # armed with the best nu, get the classifiers
    for train in datasetsTraining:
        training_set = datasetsTraining[train].values[:, cut_off:]
        training_response_set = datasetsTraining[train]["Relevant"]
        clf = NuSVC(gamma='scale', kernel='linear', nu=best_nu)
        clf.fit(training_set, labelfy(training_response_set))
        classifiers[train] = clf
        all_data = {}
        for test in datasetsTesting:
            testing_set = datasetsTesting[test].values[:, cut_off:]
            testing_response_set = datasetsTesting[test]
            # overall accuracy
            classifier_scores[(train, test)] = clf.score(testing_set, labelfy(testing_response_set["Relevant"]))
            # per stimulus accuracy
            plot_ys = []
            y_errs = []
            plt.title("{}; Measurement noise SD: {}".format(train, sd_scale))
            plt.ylabel("Accuracy")
            plt.xlabel("Presented Stimulus")
            plt.xticks(stimuli)
            for s in stimuli:
                testing_set_s = datasetsTesting[test].query("Relevant == '{}'".format(s)).values[:, cut_off:]
                testing_response_set_s = testing_response_set.query("Relevant == '{}'".format(s))["Relevant"]
                score = clf.score(testing_set_s, labelfy(testing_response_set_s))
                classifier_scores[(train, test, s)] = score
                n = testing_response_set_s.shape[0]
                mean, var = binom.stats(n, score, moments='mv')
                # Agresti–Coull interval (binomial confidence interval estimation)
                n_tilde = n + z**2
                p_tilde = 1/n_tilde * (mean + z**2/2)
                # percentage sd is the same no matter what n is present (if probability is constant).
                #sd = np.sqrt(var/n)
                #print(mean/n, sd, sd/np.sqrt(n))
                #plot_ys.append(score*100) #percent
                #plot_ys.append(mean/n)
                plot_ys.append(p_tilde)
                
                # binomial confidence interval (zSE)
                y_errs.append(z*np.sqrt(p_tilde/n_tilde * (1-p_tilde)))
            #print(train, test)
            #plt.plot(stimuli, plot_ys)
            plt.errorbar(stimuli, plot_ys, yerr=y_errs, capsize=10)
            all_data["{}_{}_{}".format(train, test, "stimuli")] = np.array(stimuli)
            all_data["{}_{}_{}".format(train, test, "accuracy")] = np.array(plot_ys)
            all_data["{}_{}_{}".format(train, test, "error")] = np.array(y_errs)

        
        acc_keys = [k for k in all_data.keys() if k.endswith("accuracy")]
        err_keys = [k for k in all_data.keys() if k.endswith("error")]

        compare = {}
        # check for statistical differences by looking for where overlaps do not occur.
        for i, key1 in enumerate(acc_keys):
            accs1 = all_data[acc_keys[i]]
            errs1 = all_data[err_keys[i]]
            for k, key2 in enumerate(acc_keys):
                dict_key = "{}_{}".format(key1, key2)
                rev_dict_key = "{}_{}".format(key2, key1)
                if k == i or rev_dict_key in compare:
                    continue
                accs2 = all_data[acc_keys[k]]
                errs2 = all_data[err_keys[k]]
                no_difference_idx = np.where(np.abs(accs1 - accs2) <= errs1+errs2)
                # default all significant
                compare[dict_key] = np.ones(accs1.shape, dtype=np.bool)
                # turn off nonsignificant indicies
                compare[dict_key][no_difference_idx] = False

                    
        plt.legend(datasetsTesting.keys())
        plt.show()
    return all_data, compare

def get_average_activities(data_file, sd_scale, alpha=0.05, cut_off = 5):
    df = pd.read_csv(data_file)
    stimuli = np.sort(np.unique(df["Relevant"].values))
    irrelevant_dims = np.sort(np.unique(df["Irrelevant"].values))
    plt.title("Average Activities; Measurement noise SD: {}".format(sd_scale))
    plt.xticks(stimuli)
    plt.xlabel("Presented Stimulus")
    plt.ylabel("'Voxel' activity")
    z = norm.ppf(1-alpha/2)
    for trained in irrelevant_dims:
        plot_ys = []
        y_errs = []
        for s in stimuli:
            m_channels = df.query("Relevant == {} and Irrelevant == '{}'".format(s, trained)).values[:, cut_off:]
            plot_ys.append(np.mean(m_channels))
            # confidence interval
            y_errs.append(z*np.std(m_channels)/np.sqrt(m_channels.size))
        plt.errorbar(stimuli, plot_ys, yerr=y_errs, capsize=10)
        
    plt.legend(irrelevant_dims)
    plt.show()

def get_idx(train, test, s, trained, tested, stimulus):
    for i in range(stimulus.shape[0]):
        if train == trained[i] and test == tested[i] and s == stimulus[i]:
            return i
    return -1

def z_compare(data_file, nu, sd_scale, alpha=0.05, cut_off = 5):
    df = pd.read_csv(data_file)
    datasetsTraining = {}
    datasetsTesting = {}
    classifiers = {}
    classifier_scores = {}
    
    training = df.query("DatasetNumber==1")
    testing = df.query("DatasetNumber==2")

    stimuli = np.sort(np.unique(training["Relevant"].values))
    irrelevant_dims = np.unique(training["Irrelevant"].values)

    for key in irrelevant_dims:
        datasetsTraining["{} training".format(key)] = training.query("Irrelevant=='{}'".format(key))
        datasetsTesting["{} testing".format(key)] = testing.query("Irrelevant=='{}'".format(key))
    
    
    combos = len(datasetsTraining.keys())*len(datasetsTesting.keys())*stimuli.shape[0]
    acc = np.zeros(combos)
    p = np.zeros(combos)
    n = np.zeros(combos)
    x = np.zeros(combos)
    trained = np.zeros(combos, dtype=object)
    tested = np.zeros(combos, dtype=object)
    stimulus = np.zeros(combos)
    ci_high = np.zeros(combos)
    ci_low = np.zeros(combos)
    i = 0

    for train in datasetsTraining:
        training_set = datasetsTraining[train].values[:, cut_off:]
        training_response_set = datasetsTraining[train]["Relevant"]
        
        clf = NuSVC(gamma='scale', kernel='linear', nu=nu)
        clf.fit(training_set, labelfy(training_response_set))

        

        for test in datasetsTesting:
            testing_set = datasetsTesting[test].values[:, cut_off:]
            testing_response_set = datasetsTesting[test]
            for s in stimuli:
                testing_set_s = datasetsTesting[test].query("Relevant == '{}'".format(s)).values[:, cut_off:]
                testing_response_set_s = testing_response_set.query("Relevant == '{}'".format(s))["Relevant"]

                acc[i] = clf.score(testing_set_s, labelfy(testing_response_set_s))
                n[i] = testing_response_set_s.shape[0]
                x[i] = int(np.round(n[i] * acc[i],0))
                p[i] = binom_test(x[i], n=n[i], p=1./stimuli.shape[0], alternative='greater')
                ci_low[i], ci_high[i] = proportion_confint(x[i],n[i])
                trained[i] = train
                tested[i] = test
                stimulus[i] = s
                i+=1

    reject,cor_p,_,_,=multipletests(p)
    print('p-values corrected using Holm-Sidak method')
    for i in range(p.shape[0]):
        print("Trained {}, Tested {}, stim {}: {}-{}; p={}, reject: {}".format(trained[i], tested[i], stimulus[i], ci_low[i], ci_high[i], cor_p[i], reject[i]))

    # make a DataFrame containing the train, test, stimulus, n, x, acc data
    dictionary = dict(trained=trained, tested=tested, x=x, n=n, acc=acc, stimulus=stimulus, ci_low=ci_low, ci_high=ci_high, p=p, cor_p=cor_p, reject=reject)
    df = pd.DataFrame(dictionary)
    df.to_csv("confidence_intervals_{}.csv".format(sd_scale))

    # pairwise trained on corresponding testA vs testB
    paircount = stimuli.shape[0] * len(datasetsTraining.keys()) * (len(datasetsTesting.keys())-1)# stimuli * irrelevant dims * irrelevant dims-1
    p_pairwise = np.zeros(paircount)
    z = np.zeros(paircount)
    xA = np.zeros(paircount)
    xB = np.zeros(paircount)
    nA = np.zeros(paircount)
    nB = np.zeros(paircount)
    trained_pairwise = np.zeros(paircount, dtype=object)
    tested_pairwise = np.zeros(paircount, dtype=object)
    stimuli_pairwise = np.zeros(paircount)
    i = 0

    for train in datasetsTraining:
        a = train.split(" ")[:-1]
        aC =("{}"*len(a[:-1])).format(*a[:-1])
        for testA in datasetsTesting:
            b = testA.split(" ")[:-1]
            bC = ("{}"*len(b[:-1])).format(*b[:-1])
            if aC != bC:
                continue
            for testB in datasetsTesting:
                if testA == testB:
                    continue
                for s in stimuli:
                    i1 = get_idx(train, testA, s, trained, tested, stimulus)
                    i2 = get_idx(train, testB, s, trained, tested, stimulus)
                    z[i],p_pairwise[i] = proportions_ztest(x[[i1, i2]], n[[i1, i2]])
                    xA[i] = x[i1]
                    xB[i] = x[i2]
                    nA[i] = n[i1]
                    nB[i] = n[i2]
                    trained_pairwise[i] = testA # also indicates the value of train
                    tested_pairwise[i] = testB
                    stimuli_pairwise[i] = s
                    i+=1
    reject_pairwise,cor_p_pairwise,_,_,=multipletests(p_pairwise)

    for i in range(p_pairwise.shape[0]):
        print("Trained {}, Tested {}, stim {}: z={}, p={}, reject: {}".format(trained_pairwise[i], tested_pairwise[i], stimuli_pairwise[i], z[i], cor_p_pairwise[i], reject_pairwise[i]))

    dictionary = dict(trained=trained_pairwise, tested=tested_pairwise, xTrained=xA, xTested=xB, nTrained=nA, nTested=nB, z=z, stimulus=stimuli_pairwise, p=p_pairwise, cor_p=cor_p_pairwise, reject=reject_pairwise)
    # dataframe containing each pairwise comparison
    df2 = pd.DataFrame(dictionary)
    df2.to_csv("pairwise_{}.csv".format(sd_scale))
    return df, df2

def gaussian_int(func, encoder, **kwargs):
    def wrap(x, noise_sigma=0.0, **kwargs):
        stimuli = [x]#+[random_stimulus(encoder.domain) for i in range(noise_stimuli)]
        responses = None#np.zeros(len(encoder.channels))
        for i,s in enumerate(stimuli):
            if kwargs:
                val = np.array(func(s, **kwargs),dtype=float).reshape(-1, len(encoder.channels))
            else:
                val = np.array(func(s),dtype=float).reshape(-1, len(encoder.channels))
            if responses is None:
                responses = np.random.normal(0, noise_sigma, size=val.shape)
            responses += val
        return responses/len(stimuli)
    return wrap
