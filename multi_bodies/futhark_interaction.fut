import "algebra"
import "box"
import "associations"
import "hgo"
import "potential"

let fromLO (location:[3]f32) (orientation:[4]f32) : (v3, quaternion) = 
    ( (location[0], location[1], location[2])
    , quaternion orientation[0] (orientation[1], orientation[2], orientation[3]) )

let fromLOs = map2 fromLO

let reshapeFTs (FTs: [](v3, v3)) : [][3]f32 =
    let toArr (x,y,z) = [x,y,z]
    in map (\(f,t) -> [toArr f, toArr t]) FTs |> flatten

let fittingBox cutoffL coords = 
    let maxDim = map (.0) coords |> reduce (triadMap2 f32.max) (0,0,0)
    let minDim = map (.0) coords |> reduce (triadMap2 f32.min) (0,0,0)
    let box = m33diag (v3add (cutoffL, cutoffL, cutoffL) (v3sub maxDim minDim))
    let (ps, os) = unzip coords
    let coords' = zip (map (`v3sub` minDim) ps) os
    in (box, coords')

entry networkPotential (parameter: networkParameter [][]) location orientation = 
    let net = fromParameter parameter 
    in (forceTorquePot net (fromLO location orientation)).1

entry hgoPotential epsilon sigma_par sigma_ort location orientation = 
    hgo epsilon sigma_par sigma_ort (fromLO location orientation)

entry networkPotentialAbs (parameter: networkParameter [][])
    location0 orientation0 
    location1 orientation1
    = 
    let net = fromParameter parameter 
    in (forceTorquePot net 
        (toRelative (fromLO location0 orientation0) (fromLO location1 orientation1))
       ).1

entry hgoPotentialAbs epsilon sigma_par sigma_ort
    location0 orientation0 
    location1 orientation1 
    = 
    hgo epsilon sigma_par sigma_ort 
        (toRelative (fromLO location0 orientation0) (fromLO location1 orientation1))

entry networkInteraction (parameter: networkParameter [][]) locations orientations =
    let coords = fromLOs locations orientations
    let net = fromParameter parameter 
    let cutoff = net.cutoff net.net.parameter
    let cutoffL = representativeLength cutoff 
    let potential = forceTorquePot net
    let (box, coords') = fittingBox cutoffL coords
    in gridInteraction potential cutoffL cutoffL (coordinateTest box coords' (inside cutoff)) box coords'
    |> (.0)
    |> reshapeFTs

entry hgoInteraction epsilon sigma_par sigma_ort locations orientations =
    let coords = fromLOs locations orientations
    let cutoffL = 3 * (sigma_par + sigma_ort)
    let rodL = sigma_par
    let cutoff = (#rods ((0,0,rodL),(0,0,rodL)), cutoffL, 1) 
    let potential = ForceTorqueAD (hgo epsilon sigma_par sigma_ort)
    let (box, coords') = fittingBox cutoffL coords
    in gridInteraction potential cutoffL cutoffL (coordinateTest box coords' (inside cutoff)) box coords'
    |> (.0)
    |> reshapeFTs
