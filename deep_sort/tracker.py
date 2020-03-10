# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from shapely.geometry  import box, Polygon, Point
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
import csv
import cv2
from .event import Event, EventState
from misc.utils import area_inside, IOU
#from .territories import P2
from .territory import Territory, TerrState
#points = [(4,22),(121,96),(207,99),(537,366),(819,324),(1564,385),(1764,322),(1648,889),(105,915),(3,762)]
points = [(2,24),(127,24),(579,321),(1746,336),(1674,878),(1912,957),(1926,1081),(2,1074)]

zone_polygon = Polygon(points)
text_count_size = 9e-3 * 200
text_count_x,text_count_y = 550,  1000


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age_chick=27000, max_age_bird=300, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age_bird = max_age_bird
        self.max_age_chick = max_age_chick
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.terr = {}
     
    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)
    '''
    def update_terr(self):

       for id,v in self.terr.items():
          v.time_since_update += 1 

       for track in self.tracks:
           if not track.is_confirmed() or track.time_since_update > 1  or track.class_id == 0 :
               continue
        
           if track.track_id in self.terr.keys(): 
               self.terr[track.track_id].update(track.mean)
               self.terr[track.track_id].time_since_update = 0 
               
           else :
               self.terr.update({track.track_id:Territory(100, track.mean, TerrState.Empty,track.track_id)})
      
       # iterate thrugh the dictionary and check if IOU>0.8 then, merge the territories
       if self.terr != {}:
           delete = []
           (k,t) = zip(*self.terr.items())
           for i in range(len(t)):
               for j in range(i+1,len(t)):
                   a=t[i].return_point_object() 
                   b=t[j].return_point_object()
                   if IOU(a,b) > 0.7:
                     if t[i].time_since_update > t[j].time_since_update:
                        t[i].update((t[j].x,t[j].y))
                        t[i].time_since_update = 0
                        delete.append(k[j])
                     else:
                        t[j].update((t[i].x,t[i].y))
                        t[j].time_since_update = 0
                        delete.append(k[i]) 
          
           for key in delete:
               self.terr.pop(key,None)       
            

    ''' 
    def update_events(self,draw_frame, timestamp, f,g, write):
        pos = []
        for i,terr in self.terr.items():
           pos = pos + [terr.x]
           pos = pos + [terr.y]
        for track in self.tracks:
            takeover_per = []
            bbox = track.to_tlbr()
            cv2.rectangle(draw_frame,(int(bbox[0]), int(bbox[1])),(int(bbox[2]),int(bbox[3])),(255,255,255),2)
            if not track.is_confirmed() or track.time_since_update > 1 :
                continue

            detection_polygon=box(int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
            z = area_inside(zone_polygon, detection_polygon)
            prev_event = track.events[-1]

            if track.hits < 5 and z < 0.75 and not prev_event.is_arrived():
                track.events.append(Event(timestamp,EventState.Arrival))
                f.writerow([timestamp,track.class_id, track.track_id, 'Arrival'])
                cv2.putText(draw_frame,"Arrival Event detected !!!! ",(text_count_x,text_count_y),4,text_count_size, (255,0,0),4)
            elif track.hits > 30 and z < 0.6 and not prev_event.is_departed():
                track.events.append(Event(timestamp,EventState.Departure))
                f.writerow([timestamp,track.class_id, track.track_id, 'Departure'])
                cv2.putText(draw_frame,"Departure Event detected !!!! ",(text_count_x,text_count_y),4,text_count_size, (255,0,0),4)
            elif z==1 and track.class_id==2 and track.hits<5 and not zone_polygon.contains(detection_polygon.centroid):
                track.events.append(Event(timestamp,EventState.Egg))
                f.writerow([timestamp,track.class_id, track.track_id, 'Egg Layed'])
                cv2.putText(draw_frame,"Egg laying Event detected !!!!",(text_count_x,text_count_y),4,text_count_size, (255,0,0),4)
            for id, terr in self.terr.items(): 
                p = area_inside( detection_polygon, terr.return_point_object())
                if  not track.class_id :
                   takeover_per = takeover_per + [p] 
                   track.update_id(terr)
                cv2.putText(draw_frame, format(p,'.2f'),(int(terr.x),int(terr.y)),0, 5e-3 * 200, (255,255,255),2)
            cv2.putText(draw_frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (255,0,0),2)
            if write:
                g.writerow([timestamp,track.track_id,track.class_id, bbox ] + pos + takeover_per)

    def update(self, detections, class_ids, timestamp):

        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], class_ids[detection_idx], timestamp)
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        #Update distance metric.
        '''active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)
        '''
    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        def gated_metric_2(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].to_xy()  for i in detection_indices])
            targets = np.array([tracks[i].to_xy() for i in track_indices])
            cost_matrix = self.metric.distance_2(features, targets)
            print('Initial:',cost_matrix)
            '''cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices, True)'''
            print('gated:',cost_matrix)
            for row, track_idx in enumerate(track_indices):
               for col, det_idx in enumerate(detection_indices):
                   if (tracks[track_idx].class_id != detections[det_idx].class_id):
                       cost_matrix[row, col] = linear_assignment.INFTY_COST

            return cost_matrix

        '''
        # Split track set into confirmed and unconfirmed tracks.
               # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)
       '''

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        '''iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]'''

        unmatched_detections = np.arange(len(detections ))
        iou_track_candidates = np.arange(len(self.tracks))
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)
        
        confirmed_tracks = [
            i for i in unmatched_tracks_b  if self.tracks[i].is_confirmed()]
        unconfirmed_tracks = [
            i for i in unmatched_tracks_b if not self.tracks[i].is_confirmed()]


        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.min_cost_matching(
                gated_metric_2, self.metric.matching_threshold,
                self.tracks, detections, confirmed_tracks, unmatched_detections)
     

        '''matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks, unmatched_detections)
        '''


       #Associate remaining tracks with the apperance model
       #1. have to get features for the particular detections
       #2. See if they match with the tracks.
       #3. 
        
        matches = matches_b + matches_a
        unmatched_tracks = list(set( unmatched_tracks_a + unconfirmed_tracks))
        return matches, unmatched_tracks, unmatched_detections


    def _match_old(self, detections):
   
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

   
    def _initiate_track(self, detection, class_id, timestamp):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        if class_id != 0 :
            self.tracks.append(Track(
            mean, covariance, self._next_id, class_id, timestamp, self.n_init, self.max_age_chick,
            detection.feature))
        else:
            self.tracks.append(Track(
            mean, covariance, self._next_id, class_id, timestamp, self.n_init, self.max_age_bird,
            detection.feature))
        self._next_id += 1
