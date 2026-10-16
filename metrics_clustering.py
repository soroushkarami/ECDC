def count_ch_changes(prev_ch: dict, curr_ch: dict):
    ch_changes = 0
    for cluster_id in curr_ch:
        if cluster_id in prev_ch:
            if prev_ch[cluster_id] != curr_ch[cluster_id]:
                ch_changes += 1
    return ch_changes


# each zone can have at most one cluster; a zone has a cluster if it has an elected CH
def count_cl_changes(prev_zone_members: dict, curr_zone_members: dict):
    cl_changes = 0
    for zone in curr_zone_members:
        if zone in prev_zone_members:  # if the zone had a cluster before
            if set(prev_zone_members[zone]) != set(curr_zone_members[zone]):
                cl_changes += 1
    return cl_changes


def lifetime(all_lifetimes: dict[str, list[int]], cluster_birth: dict, died_clusters: set, curr_step: float):
    for cl in died_clusters:
        if cl in cluster_birth:
            # the cluster is dead so compute its lifetime
            cl_lifetime = curr_step - cluster_birth.pop(cl)  # pop() to remove the birth time.
            if cl not in all_lifetimes:
                all_lifetimes[cl] = []
            all_lifetimes[cl].append(cl_lifetime)
    return all_lifetimes


def track_zone_lifetimes(curr_zone_members: dict, birth_time: dict, lifetimes: list, curr_step: float) -> tuple[dict, list]:
    """
    Update zone (cluster) lifetimes based on whether a zone has members (active+dead).
    Returns updated (birth_time, lifetimes).

    Parameters:
        curr_zone_members: dict mapping zone_id -> set of member IDs (both active and dead)
        birth_time: dict mapping zone_id -> time when zone became non-empty
        lifetimes: list to append completed lifetimes
        curr_step: current simulation time (seconds)

    Returns:
        (birth_time, lifetimes) after processing.
    """
    for zone_id, members in curr_zone_members.items():
        if members:  # zone is non-empty
            if zone_id not in birth_time:
                birth_time[zone_id] = curr_step
        else:  # zone is empty
            if zone_id in birth_time:
                life = curr_step - birth_time.pop(zone_id)
                lifetimes.append(life)
    return birth_time, lifetimes


def track_membership_lifetimes(prev_members: dict, curr_members: dict, birth_time: dict, lifetimes: list, curr_step: float) -> tuple[dict, list]:
    """
    Track cluster lifetimes based on membership set changes.
    A cluster's lifetime ends when its member set changes (any addition/removal).
    Returns updated (birth_time, lifetimes).
    """
    # Process zones that are currently present
    for zone_id, curr_set in curr_members.items():
        prev_set = prev_members.get(zone_id, set())
        if curr_set != prev_set:
            # Membership changed: end previous cluster if it existed
            if zone_id in birth_time:
                life = curr_step - birth_time.pop(zone_id)
                lifetimes.append(life)
            # Start a new cluster if the new set is non-empty
            if curr_set:
                birth_time[zone_id] = curr_step
    # Process zones that disappeared (in prev but not in curr)
    for zone_id in set(prev_members.keys()) - set(curr_members.keys()):
        if zone_id in birth_time:
            life = curr_step - birth_time.pop(zone_id)
            lifetimes.append(life)
    return birth_time, lifetimes


def baseline_style_metrics(prev_members: dict, curr_members: dict,
                           birth_time: dict, lifetimes: list,
                           curr_step: float) -> tuple[dict, list, int]:
    """
    Update cluster lifetimes and count membership changes exactly as in baseline.
    Returns (birth_time, lifetimes, cluster_changes_this_step)
    """
    cluster_changes = 0

    # Process zones that exist in both steps
    for zone_id in set(prev_members.keys()) & set(curr_members.keys()):
        if prev_members[zone_id] != curr_members[zone_id]:
            # Membership changed -> end previous lifetime
            if zone_id in birth_time:
                lifetimes.append(curr_step - birth_time.pop(zone_id))
            # Start new lifetime if non-empty
            if curr_members[zone_id]:
                birth_time[zone_id] = curr_step
            cluster_changes += 1

    # Process zones that disappeared (in prev but not in curr)
    for zone_id in set(prev_members.keys()) - set(curr_members.keys()):
        if zone_id in birth_time:
            lifetimes.append(curr_step - birth_time.pop(zone_id))
        # Do NOT increment cluster_changes here (matches baseline)

    # Process zones that appeared (in curr but not in prev)
    for zone_id in set(curr_members.keys()) - set(prev_members.keys()):
        if curr_members[zone_id]:
            birth_time[zone_id] = curr_step

    return birth_time, lifetimes, cluster_changes


# --------------------- Additional metrics for hypothesis testing ---------------------

def track_cluster_births(curr_zone_members: dict, prev_zone_members: dict,
                         birth_counter: dict, time_step: float) -> dict:
    """
    Count how many new clusters are born in this step.
    A cluster is born if it was empty in prev and non-empty in curr.
    """
    for zone_id, members in curr_zone_members.items():
        if members:
            prev_set = prev_zone_members.get(zone_id, set())
            if not prev_set:
                # cluster born
                birth_counter[zone_id] = birth_counter.get(zone_id, 0) + 1
    return birth_counter

def track_cluster_size(curr_zone_members: dict, size_list: list) -> list:
    """
    Record the size (number of members) of each active cluster at this step.
    Returns updated size_list (appended with average size this step).
    """
    sizes = [len(members) for members in curr_zone_members.values() if members]
    if sizes:
        size_list.append(sum(sizes) / len(sizes))  # average size this step
    return size_list

def track_ch_tenure(curr_ch_map: dict, prev_ch_map: dict, ch_start_time: dict,
                    ch_end_time: dict, current_time: float) -> tuple[dict, dict]:
    """
    Track how long each vehicle stays as CH.
    - ch_start_time: dict {vehicle_id: start_time}
    - ch_end_time: dict {vehicle_id: end_time}
    When a vehicle ceases to be CH, record its end time.
    """
    # Record start time for new CHs
    for cluster_id, ch_veh in curr_ch_map.items():
        if ch_veh not in ch_start_time:
            ch_start_time[ch_veh] = current_time

    # Check which CHs are no longer CH
    prev_ch_set = set(prev_ch_map.values())
    curr_ch_set = set(curr_ch_map.values())
    for ch_veh in prev_ch_set - curr_ch_set:
        if ch_veh in ch_start_time and ch_veh not in ch_end_time:
            ch_end_time[ch_veh] = current_time

    return ch_start_time, ch_end_time

def compute_ch_tenure_histogram(ch_start_time: dict, ch_end_time: dict,
                                current_time: float) -> list:
    """
    Return a list of CH tenures (in seconds) for vehicles that have ended their CH role.
    If a vehicle is still CH at the end, we can optionally compute tenure until end of sim.
    """
    tenures = []
    for veh, end in ch_end_time.items():
        if veh in ch_start_time:
            start = ch_start_time[veh]
            tenures.append(end - start)
    # Optionally, for CHs still active at current_time, add a final tenure at simulation end
    # We'll let the simulation loop handle that at the end.
    return tenures