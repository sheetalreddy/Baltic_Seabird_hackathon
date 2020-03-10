# vim: expandtab:ts=4:sw=4
import numpy as np
from shapely.geometry import Point

class TerrState:

    Empty = 0
    Occupied = 1
    Tentative = 2
    Update = 3
   
class Territory:

    def __init__(self,radius,p,state,track_id):
        self.radius = radius
        self.hits = 1
        self.age = 1
        self.track_id = track_id
        self.x = p[0]
        self.y = p[1]
        self.time_since_update = 0
        self.state = state
        self._n_init = 75 # 5 sec 
        self._max_age = 100000000 #the territory needs a position update if time_since_update > max_age

    '''def update_id(self, track ):

        #self.time_since_update = 0
        takeover = 0
        if self.track_id != track.track_id :
            self.n_hits = self.n_hits + 1
            if self.n_hits > 15: 
                # n_hits is the no of times the territory is takenover, if its greater than 1 sec(not consecutive)
                self.track_id = track.track_id
                self.state = TerrState.Tentative
                self.hits = 1
                self.n_hits = 0 
        else : 
        self.hits += 1
        if self.state == TerrState.Tentative and self.hits >= self._n_init:
            self.state = TerrState.Occupied
            self.terr_h = self.track_id
            takeover = 1 
        elif self.state == TerrState.Tentative and self.hits <= self._n_init:
            self.terr_h = 0
        elif self.state == TerrState.Empty:
            self.state = TerrState.Tentative

        return takeover
    '''
    def update(self, point ):
         self.x = point[0]
         self.y = point[1]
         self.time_since_update = 0 

    def return_point_object(self):
        return Point(self.x,self.y).buffer(self.radius)


    def mark_missed(self):
        delete = 0 
        self.time_since_update +=1 
        if self.time_since_update > self._max_age:
            delete  = 1
        return delete  

    def is_occupied(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TerrState.Occupied

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TerrState.Tentative

    def is_empty(self):
        """Returns True if this track is confirmed."""
        return self.state == TerrState.Empty
  
