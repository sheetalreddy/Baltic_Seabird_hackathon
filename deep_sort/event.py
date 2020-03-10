# vim: expandtab:ts=4:sw=4


class EventState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """
    Seen = 0 
    Arrival = 1
    Paired = 2
    Egg = 3
    Takeover = 4
    Departure = 5


class Event:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, timestamp,state, n_init=1, max_age=1):
        self.timestamp = timestamp
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = state
        self._n_init = n_init
        self._max_age = max_age

    '''def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == EventState.Tentative and self.hits >= self._n_init:
            self.state = EventState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted'''

    def is_arrived(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == EventState.Arrival

    def is_seen(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == EventState.Seen

    def is_paired(self):
        """Returns True if this track is confirmed."""
        return self.state == EventState.Paired

    def is_takeover(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state ==  EventState.Takeover

    def is_departed(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state ==  EventState.Departure

