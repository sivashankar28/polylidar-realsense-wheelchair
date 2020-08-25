What did we need done in tracking.py?

# Predict Curb Height

1. Extract vertices (points) from each polygon. Each polygon will also have an "approximate" plane normal associated with it.
2. Perform RANSAC plane fitting to all these reduced point sets. Geometric planes will be returned.
3. Determine curb height from geometric plane normals
    1. Find the pair of geometric planes representing the curb and the road. (should have same normal roughly)
    2. Compute plane to plane distance. This is the curb height.


# Predict the yaw deviation between wheel chair frame and the curb

This can be predicted in one of three ways


1. Extract the curb-height surface. Compute its geometric plane to get normal n_{ch} in sensor frame. Relate this to the body frame and compute yaw deviation.
2. Find the separating line between the curb and street level surface. This line is computed using SVM with the projection of curb and street point sets. Relate this line to the body frame and compute yaw deviation.



# Predict the pitch deviation between wheel chair and incoming incline plane

1. Extract the vertices of the inclined plane
2. Perform RANSAC plane fitting to all these reduced point sets. Geometric planes will be returned.
3. Use incline plane normal to compute pitch.