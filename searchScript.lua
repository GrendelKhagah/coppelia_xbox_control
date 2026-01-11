-- searchScript searches for a /bill entity in the simulation
function sysCall_actuation()
    local dt = sim.getSimulationTimeStep()
    local wantSearch = ((sim.getInt32Signal("SEARCH_ON") or 0) == 1)

    if not wantSearch then
        searchOn = false
        avoidActive = false
        foundHold = 0.0
        return
    end

    -- Ensure bill is in the sim.
    if victim == -1 then
        sim.addLog(sim.verbosity_errors, "[SEARCH] SEARCH_ON=1 but /Bill not found. Disabling.")
        sim.setInt32Signal("SEARCH_ON", 0)
        searchOn = false
        return
    end

    -- Start search (one-time init)
    if wantSearch and not searchOn then
        searchOn = true
        avoidActive = false
        foundHold = 0.0

        -- Use current target altitude, fallback
        local ok, pos = pcall(sim.getObjectPosition, target, -1)
        if ok and pos and #pos >= 3 and pos[3] ~= nil then
            SEARCH_ALT = pos[3]
        else
            SEARCH_ALT = SEARCH_ALT_DEFAULT
        end

        local bpos = sim.getObjectPosition(base, -1)
        wp = buildLawnmower({bpos[1], bpos[2]}, WIDTH, HEIGHT, LANE, SEARCH_ALT)
        wp_i = 1

        -- Match yaw to drone
        local bor = sim.getObjectOrientation(base, -1)
        yawSet = bor[3]
        sim.setObjectOrientation(target, -1, {0.0, 0.0, yawSet})

        sim.addLog(sim.verbosity_scriptinfos, "[SEARCH] started (simple avoidance)")
    end

    -- is the victim found? if yes stop search and declare target found
    if foundVictim(dt) then
        sim.addLog(sim.verbosity_scriptinfos, "[SEARCH] FOUND target! Disabling SEARCH_ON.")
        sim.setInt32Signal("SEARCH_ON", 0)
        searchOn = false
        return
    end

    if #wp == 0 then return end

    -- Current target position
    local tpos = sim.getObjectPosition(target, -1)

    -- Altimeter safety: if too low, climb (no XY move)
    local altHit, altDist = readAlt()
    if altHit and altDist < GROUND_MARGIN then
        local desiredZ = tpos[3] + CLIMB_SPEED * dt
        desiredZ = math.max(desiredZ, SEARCH_ALT + (GROUND_MARGIN - altDist) + CLIMB_BONUS)
        sim.setObjectPosition(target, -1, {tpos[1], tpos[2], desiredZ})
        return
    end

    -- Waypoint goal (XY)
    local goal = {wp[wp_i][1], wp[wp_i][2], SEARCH_ALT}

    -- Yaw steer toward goal
    steerYawTowardGoal(dt, {goal[1], goal[2]})

    -- read sensors
    local frontHit, frontDist = readFront()
    local leftTriHit, leftTriDist = readTri(left_tri)
    local rightTriHit, rightTriDist = readTri(right_tri)
    local flTriHit, flTriDist = readTri(front_left_tri)
    local frTriHit, frTriDist = readTri(front_right_tri)

    -- Triangle blocks
    local frontTriBlock =
        (flTriHit and flTriDist <= TRI_FRONT_STOP_DIST) or
        (frTriHit and frTriDist <= TRI_FRONT_STOP_DIST)
    local leftBlock  = (leftTriHit  and leftTriDist  <= TRI_SIDE_STOP_DIST)
    local rightBlock = (rightTriHit and rightTriDist <= TRI_SIDE_STOP_DIST)

    -- Front blocks: either triangle front, or front sensor STOP_DIST
    local frontBlock = frontTriBlock or (frontHit and frontDist <= STOP_DIST)

    -- if blocked and not already avoiding, pick a sidestep direction and start timer
    if frontBlock and not avoidActive then
        -- Prefer stepping away from a blocked side; otherwise pick away from nearer side
        if leftBlock and not rightBlock then
            avoidDir = 1  -- go RIGHT
        elseif rightBlock and not leftBlock then
            avoidDir = -1 -- go LEFT
        else
            -- choose the side that seems "more open"
            local l = (leftTriHit and leftTriDist) or math.huge
            local r = (rightTriHit and rightTriDist) or math.huge
            avoidDir = (r > l) and 1 or -1
        end

        avoidActive = true
        avoidT = AVOID_TIME
        sim.addLog(sim.verbosity_scriptinfos, "[SEARCH] blocked -> sidestep " .. (avoidDir == 1 and "RIGHT" or "LEFT"))
    end

    -- Avoidance step (pure sidestep for a short time)
    if avoidActive then
        avoidT = avoidT - dt

        -- If the side we want is blocked, stop (prevents sliding into something)
        if (avoidDir == -1 and leftBlock) or (avoidDir == 1 and rightBlock) then
            avoidActive = false
        else
            local tor = sim.getObjectOrientation(target, -1)
            local yaw = tor[3]

            -- Strafe vector in world frame (right = yaw+90)
            local sx = math.cos(yaw + math.pi/2) * (AVOID_SPEED * dt * avoidDir)
            local sy = math.sin(yaw + math.pi/2) * (AVOID_SPEED * dt * avoidDir)

            sim.setObjectPosition(target, -1, {tpos[1] + sx, tpos[2] + sy, SEARCH_ALT})
        end

        if avoidT <= 0.0 then
            avoidActive = false
        end
        return
    end

    -- Speed scaling based on front sensor (simple slow/stop)
    local speedScale = 1.0
    if frontBlock then
        speedScale = 0.0
    elseif frontHit then
        if frontDist < SLOW_DIST then
            speedScale = (frontDist - STOP_DIST) / math.max((SLOW_DIST - STOP_DIST), 1e-6)
            speedScale = clamp(speedScale, 0.0, 1.0)
        end
    end

    -- Move /target gradually toward the waypoint
    local maxStep = TARGET_SPEED * speedScale * dt
    maxStep = math.min(maxStep, TARGET_MAX_STEP)

    if maxStep > 1e-6 then
        local newPos, _ = moveToward({tpos[1], tpos[2], SEARCH_ALT}, goal, maxStep)
        sim.setObjectPosition(target, -1, newPos)
    else
        sim.setObjectPosition(target, -1, {tpos[1], tpos[2], SEARCH_ALT})
    end

    -- Advance waypoint when drone base reaches the goal
    local bpos = sim.getObjectPosition(base, -1)
    if dist2({bpos[1], bpos[2], SEARCH_ALT}, goal) < (WP_TOL*WP_TOL) then
        wp_i = wp_i + 1
        if wp_i > #wp then
            sim.addLog(sim.verbosity_scriptinfos, "[SEARCH] completed coverage. Disabling SEARCH_ON.")
            sim.setInt32Signal("SEARCH_ON", 0)
            searchOn = false
        end
    end
end
