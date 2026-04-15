"""
Item Tracker - Tracks items with IDs and collaboration during simulation.

Manages item lifecycle (creation, pickup, drop, cutting, assembly, delivery)
and tracks which agents touched each item for collaboration analysis.

GAME STATE SYNCHRONIZATION:
============================
The tracker now syncs with actual game state before each action to prevent
desynchronization. Instead of guessing from action names, it reads:
- game.gameObjects[agent_id].item - what each agent is holding
- game.grid.tiles[x][y].item - what's on each counter

This eliminates the shadow tracker problem where parallel state gets out of sync.

ITEM LINEAGE:
=============
Maintains complete parent-child relationships:
- tomato → tomato_cut (via cutting)
- plate + tomato_cut → tomato_salad (via assembly)

All items track their origins dictionary to maintain the full chain.

Author: Samuel Lozano
"""

from typing import Dict, List, Optional, Tuple, Any


class ItemTracker:
    """Track items with IDs and collaboration during simulation."""
    
    def __init__(self):
        self.item_counters = {'tomato': 0, 'plate': 0, 'tomato_cut': 0, 'tomato_salad': 0}
        self.items = {}  # item_id -> {type, touched_by, origins, roles}
        self.recorded_items: List[Dict[str, Any]] = []
        self._creation_counter = 0
        self._recorded_item_counter = 0
        self.agent_holding = {}  # agent_id -> item_id
        self.counter_items = {}  # (x, y) -> item_id
        
    def create_item(
        self,
        item_type: str,
        agent_id: str,
        origins: Dict = None,
        tick: Optional[int] = None,
        second: Optional[float] = None,
        source: str = '',
    ) -> str:
        """Create new tracked item."""
        self.item_counters[item_type] += 1
        item_id = f"{item_type}_{self.item_counters[item_type]}"
        self._creation_counter += 1

        # Sanitise origins: strip None / empty-string values so downstream
        # code can always treat a missing key as "unknown".
        clean_origins = {k: v for k, v in (origins or {}).items() if v}

        self.items[item_id] = {
            'type': item_type,
            'creation_index': self._creation_counter,
            'created_by': agent_id,
            'created_tick': tick if tick is not None else '',
            'created_second': f"{second:.3f}" if second is not None else '',
            'created_source': source,
            'touched_by': [agent_id] if agent_id else [],
            'last_touched': agent_id,
            'origins': clean_origins,
            'counters_used': [],
            'who_picked_tomato': '',
            'who_picked_plate': '',
            'who_cut': '',
            'who_assembled': '',
            'who_delivered': ''
        }

        # Set roles based on type and propagate from origin items
        if item_type == 'tomato' and agent_id:
            self.items[item_id]['who_picked_tomato'] = agent_id
        elif item_type == 'plate' and agent_id:
            self.items[item_id]['who_picked_plate'] = agent_id
        elif item_type == 'tomato_cut' and agent_id:
            self.items[item_id]['who_cut'] = agent_id
            # Inherit who_picked_tomato from the source tomato (if tracked)
            tomato_origin = clean_origins.get('tomato_id')
            if tomato_origin and tomato_origin in self.items:
                self.items[item_id]['who_picked_tomato'] = self.items[tomato_origin]['who_picked_tomato']
        elif item_type == 'tomato_salad' and agent_id:
            self.items[item_id]['who_assembled'] = agent_id
            # Inherit all per-step roles from origin items
            for origin_key in ['tomato_cut_id', 'plate_id']:
                origin_id = clean_origins.get(origin_key)
                if origin_id and origin_id in self.items:
                    origin = self.items[origin_id]
                    for role in ['who_picked_tomato', 'who_picked_plate', 'who_cut']:
                        if origin.get(role) and not self.items[item_id][role]:
                            self.items[item_id][role] = origin[role]
                    # Also walk one level deeper: tomato_cut -> tomato
                    if origin_key == 'tomato_cut_id':
                        grandparent_id = origin.get('origins', {}).get('tomato_id')
                        if grandparent_id and grandparent_id in self.items:
                            gp = self.items[grandparent_id]
                            if gp.get('who_picked_tomato') and not self.items[item_id]['who_picked_tomato']:
                                self.items[item_id]['who_picked_tomato'] = gp['who_picked_tomato']
        return item_id
    
    def track_pickup(self, agent_id: str, item_name: str, location: Tuple = None, from_cutting: bool = False) -> str:
        """Track pickup action."""
        if from_cutting:
            # Transform held tomato into tomato_cut
            old_item_id = self.agent_holding.get(agent_id)
            if not old_item_id:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Cutting error: {agent_id} has no item to cut!")
                # This is a critical error - create orphan to maintain consistency
                old_item_id = self.create_item('tomato', agent_id)
            item_id = self.create_item('tomato_cut', agent_id, {'tomato_id': old_item_id})
        elif location and location in self.counter_items:
            # Pick up existing item from counter
            item_id = self.counter_items.pop(location)
            if item_id not in self.items:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Counter tracking error: item {item_id} at {location} not in items dict")
                # Create new item as fallback
                item_id = self.create_item(item_name, agent_id)
            else:
                # Mark that this agent touched the item
                if agent_id not in self.items[item_id]['touched_by']:
                    self.items[item_id]['touched_by'].append(agent_id)
                self.items[item_id]['last_touched'] = agent_id
        else:
            # Pick up new item from dispenser
            item_id = self.create_item(item_name, agent_id)
        
        self.agent_holding[agent_id] = item_id
        return item_id
    
    def track_drop(self, agent_id: str, location: Tuple) -> Optional[str]:
        """Track drop action."""
        item_id = self.agent_holding.get(agent_id)
        if item_id and location:
            self.items[item_id]['counters_used'].append((location, agent_id, 'drop'))
            self.counter_items[location] = item_id
        self.agent_holding[agent_id] = None
        return item_id
    
    def track_delivery(self, agent_id: str) -> Optional[str]:
        """Track delivery."""
        item_id = self.agent_holding.get(agent_id)
        if item_id:
            self.items[item_id]['who_delivered'] = agent_id
        self.agent_holding[agent_id] = None
        return item_id

    def record_delivered_item(
        self,
        agent_id: str,
        salad_item_id: Optional[str],
        tick: Optional[int] = None,
        second: Optional[float] = None,
    ) -> Optional[str]:
        """Record a delivered item only for export, without mutating live game state."""
        if not salad_item_id or salad_item_id not in self.items:
            return None

        salad_item = self.items[salad_item_id]
        salad_data = self.get_item_data(salad_item_id)

        self._recorded_item_counter += 1
        delivered_id = f"tomato_delivered_{self._recorded_item_counter}"
        self._creation_counter += 1

        touched_list = list(salad_item.get('touched_by', []))
        if agent_id and agent_id not in touched_list:
            touched_list.append(agent_id)

        self.recorded_items.append({
            'creation_index': self._creation_counter,
            'item_id': delivered_id,
            'item_type': 'tomato_delivered',
            'created_by': agent_id,
            'created_tick': tick if tick is not None else '',
            'created_second': f"{second:.3f}" if second is not None else '',
            'created_source': 'delivery',
            'last_touched': agent_id,
            'touched_list': ';'.join(touched_list),
            'tomato_id': salad_data.get('tomato_id', ''),
            'plate_id': salad_data.get('plate_id', ''),
            'tomato_cut_id': salad_data.get('tomato_cut_id', ''),
            'tomato_salad_id': salad_item_id,
            'tomato_delivered_id': delivered_id,
            'who_picked_tomato': salad_data.get('who_picked_tomato', ''),
            'who_picked_plate': salad_data.get('who_picked_plate', ''),
            'who_cutted': salad_data.get('who_cutted', ''),
            'who_assembled': salad_data.get('who_assembled', ''),
            'who_delivered': agent_id,
            'number_of_counters_used': salad_data.get('number_of_counters_used', 0),
        })
        return delivered_id

    def track_assembly(self, agent_id: str, location: Tuple,
                       agent_item_hint: str = '') -> str:
        """Track salad assembly at a counter (action_type == 'salad_assembly').

        The agent was holding one item (plate or tomato_cut) and the counter had
        the complementary item. Both are consumed and a tomato_salad is placed
        on the counter. The triggering agent is recorded as who_assembled.

        agent_item_hint: 'plate' or 'tomato_cut' — inferred from action_name.
        """
        agent_item_id = self.agent_holding.get(agent_id)
        counter_item_id = self.counter_items.get(location)

        plate_id = None
        tomato_cut_id = None

        # Identify plate and tomato_cut from agent and counter items
        if agent_item_id and agent_item_id in self.items:
            agent_type = self.items[agent_item_id]['type']
            if agent_type == 'plate':
                plate_id = agent_item_id
            elif agent_type == 'tomato_cut':
                tomato_cut_id = agent_item_id
        
        if counter_item_id and counter_item_id in self.items:
            counter_type = self.items[counter_item_id]['type']
            if counter_type == 'plate':
                plate_id = counter_item_id
            elif counter_type == 'tomato_cut':
                tomato_cut_id = counter_item_id

        # If we still don't have both items, this is a tracking error
        # Log it but continue to maintain CSV consistency
        if plate_id is None or tomato_cut_id is None:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Assembly tracking error for {agent_id} at {location}: "
                f"plate_id={plate_id}, tomato_cut_id={tomato_cut_id}, "
                f"agent_item={agent_item_id}, counter_item={counter_item_id}, "
                f"hint={agent_item_hint}"
            )
            # Create minimal items to maintain consistency, but mark the issue
            if plate_id is None:
                plate_id = self.create_item('plate', agent_id)
                logger.warning(f"Created emergency plate: {plate_id}")
            if tomato_cut_id is None:
                tomato_cut_id = self.create_item('tomato_cut', agent_id)
                logger.warning(f"Created emergency tomato_cut: {tomato_cut_id}")

        # Clear agent holding and remove counter item at the assembly location
        self.agent_holding[agent_id] = None
        self.counter_items.pop(location, None)

        # Create salad with proper lineage
        origins = {'plate_id': plate_id, 'tomato_cut_id': tomato_cut_id}
        salad_id = self.create_item('tomato_salad', agent_id, origins)
        # Place salad on counter so a subsequent pick_up finds it
        self.counter_items[location] = salad_id
        return salad_id
    
    def get_item_data(self, item_id: str) -> Dict:
        """Get comprehensive item data for CSV output."""
        if not item_id or item_id not in self.items:
            return self._empty_data()
        
        item = self.items[item_id]
        
        # Build ID lineage — always use '' not None for missing links
        ids = {'tomato_id': '', 'plate_id': '', 'tomato_cut_id': '', 'tomato_salad_id': ''}
        if item['type'] == 'tomato':
            ids['tomato_id'] = item_id
        elif item['type'] == 'plate':
            ids['plate_id'] = item_id
        elif item['type'] == 'tomato_cut':
            ids['tomato_cut_id'] = item_id
            ids['tomato_id'] = item['origins'].get('tomato_id') or ''
        elif item['type'] == 'tomato_salad':
            ids['tomato_salad_id'] = item_id
            ids['tomato_cut_id'] = item['origins'].get('tomato_cut_id') or ''
            ids['plate_id'] = item['origins'].get('plate_id') or ''
            # Walk back through tomato_cut to find the source tomato
            tc_id = ids['tomato_cut_id']
            if tc_id and tc_id in self.items:
                ids['tomato_id'] = self.items[tc_id]['origins'].get('tomato_id') or ''
        # Collaboration tracking:
        # - item_touched are touches on this concrete item only
        # - history_touched are touches across full lineage (origins + this item)
        item_touched = set(item['touched_by'])
        history_touched_list = self._get_lineage_touched_list(item_id)
        history_touched = set(history_touched_list)
        
        return {
            **ids,
            'item_id': item_id,
            'item_type': item['type'],
            'creation_index': item.get('creation_index', ''),
            'created_by': item.get('created_by', ''),
            'created_tick': item.get('created_tick', ''),
            'created_second': item.get('created_second', ''),
            'created_source': item.get('created_source', ''),
            'last_touched': item['last_touched'],
            'touched_list': ';'.join(history_touched_list),
            'is_item_collaboration': len(item_touched) > 1,
            'is_exchange_collaboration': len(history_touched) > 1,
            'who_picked_tomato': item['who_picked_tomato'],
            'who_picked_plate': item['who_picked_plate'],
            'who_cutted': item['who_cut'],
            'who_assembled': item['who_assembled'],
            'who_delivered': item['who_delivered'],
            'number_of_counters_used': len(item['counters_used'])
        }

    def _get_lineage_touched_list(self, item_id: str) -> List[str]:
        """Return ordered unique touched agents across full item lineage."""
        if item_id not in self.items:
            return []

        # Build lineage item IDs ordered by creation index: origins first, then item.
        lineage_ids = [oid for oid in self._get_all_origins(item_id) if oid in self.items]
        lineage_ids.append(item_id)
        lineage_ids = sorted(
            set(lineage_ids),
            key=lambda iid: self.items.get(iid, {}).get('creation_index', 0)
        )

        touched_history: List[str] = []
        seen_agents = set()
        for iid in lineage_ids:
            for agent_id in self.items[iid].get('touched_by', []):
                if agent_id and agent_id not in seen_agents:
                    seen_agents.add(agent_id)
                    touched_history.append(agent_id)

        return touched_history

    def export_items(self, csv_path):
        """Write the current item ledger to items.csv."""
        import csv

        header = [
            'item_id', 'item_type', 'creation_index', 'created_by', 'created_tick', 'created_second', 'created_source',
            'last_touched', 'touched_list',
            'tomato_id', 'plate_id', 'tomato_cut_id', 'tomato_salad_id', 'tomato_delivered_id',
            'who_picked_tomato', 'who_picked_plate', 'who_cutted', 'who_assembled', 'who_delivered',
            'number_of_counters_used',
        ]

        rows = []
        for item_id, item in self.items.items():
            data = self.get_item_data(item_id)
            rows.append((item.get('creation_index', 0), [
                data.get('item_id', item_id),
                data.get('item_type', item.get('type', '')),
                data.get('creation_index', ''),
                data.get('created_by', ''),
                data.get('created_tick', ''),
                data.get('created_second', ''),
                data.get('created_source', ''),
                data.get('last_touched', ''),
                data.get('touched_list', ''),
                data.get('tomato_id', ''),
                data.get('plate_id', ''),
                data.get('tomato_cut_id', ''),
                data.get('tomato_salad_id', ''),
                data.get('tomato_delivered_id', ''),
                data.get('who_picked_tomato', ''),
                data.get('who_picked_plate', ''),
                data.get('who_cutted', ''),
                data.get('who_assembled', ''),
                data.get('who_delivered', ''),
                data.get('number_of_counters_used', 0),
            ]))

        for record in self.recorded_items:
            rows.append((record.get('creation_index', 0), [
                record.get('item_id', ''),
                record.get('item_type', ''),
                record.get('creation_index', ''),
                record.get('created_by', ''),
                record.get('created_tick', ''),
                record.get('created_second', ''),
                record.get('created_source', ''),
                record.get('last_touched', ''),
                record.get('touched_list', ''),
                record.get('tomato_id', ''),
                record.get('plate_id', ''),
                record.get('tomato_cut_id', ''),
                record.get('tomato_salad_id', ''),
                record.get('tomato_delivered_id', ''),
                record.get('who_picked_tomato', ''),
                record.get('who_picked_plate', ''),
                record.get('who_cutted', ''),
                record.get('who_assembled', ''),
                record.get('who_delivered', ''),
                record.get('number_of_counters_used', 0),
            ]))

        rows.sort(key=lambda entry: entry[0])

        with open(csv_path, 'w', newline='') as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(header)
            for _, row in rows:
                writer.writerow(row)

    def annotate_item(
        self,
        item_id: str,
        *,
        created_by: Optional[str] = None,
        created_tick: Optional[int] = None,
        created_second: Optional[float] = None,
        created_source: Optional[str] = None,
        origins: Optional[Dict[str, str]] = None,
        touched_by: Optional[str] = None,
    ) -> None:
        """Backfill item metadata discovered after sync."""
        if not item_id or item_id not in self.items:
            return

        item = self.items[item_id]
        if created_by and not item.get('created_by'):
            item['created_by'] = created_by
        if created_tick is not None and item.get('created_tick', '') == '':
            item['created_tick'] = created_tick
        if created_second is not None and item.get('created_second', '') == '':
            item['created_second'] = f"{created_second:.3f}"
        if created_source and not item.get('created_source'):
            item['created_source'] = created_source

        if origins:
            item['origins'].update({k: v for k, v in origins.items() if v and v != item_id})

        if touched_by:
            if touched_by not in item['touched_by']:
                item['touched_by'].append(touched_by)
            item['last_touched'] = touched_by

        if item['type'] == 'tomato_cut':
            tomato_origin = item['origins'].get('tomato_id')
            if tomato_origin and tomato_origin in self.items:
                source_item = self.items[tomato_origin]
                if source_item.get('who_picked_tomato') and not item.get('who_picked_tomato'):
                    item['who_picked_tomato'] = source_item['who_picked_tomato']
        elif item['type'] == 'tomato_salad':
            tomato_cut_id = item['origins'].get('tomato_cut_id')
            plate_id = item['origins'].get('plate_id')
            if tomato_cut_id and tomato_cut_id in self.items:
                source_item = self.items[tomato_cut_id]
                for source_agent in source_item.get('touched_by', []):
                    if source_agent and source_agent not in item['touched_by']:
                        item['touched_by'].insert(0, source_agent)
                for role in ['who_picked_tomato', 'who_cut']:
                    if source_item.get(role) and not item.get(role):
                        item[role] = source_item[role]
            if plate_id and plate_id in self.items:
                source_item = self.items[plate_id]
                for source_agent in source_item.get('touched_by', []):
                    if source_agent and source_agent not in item['touched_by']:
                        item['touched_by'].insert(0, source_agent)
                if source_item.get('who_picked_plate') and not item.get('who_picked_plate'):
                    item['who_picked_plate'] = source_item['who_picked_plate']
            # If no explicit touched_by update was passed, keep a meaningful
            # last_touched from ingredient lineage rather than leaving it blank.
            if not item.get('last_touched'):
                for source_id in [tomato_cut_id, plate_id]:
                    if source_id and source_id in self.items and self.items[source_id].get('last_touched'):
                        item['last_touched'] = self.items[source_id]['last_touched']
                        break
        elif item['type'] == 'tomato_delivered':
            salad_id = item['origins'].get('tomato_salad_id')
            if salad_id and salad_id in self.items:
                source_item = self.items[salad_id]
                for role in ['who_picked_tomato', 'who_picked_plate', 'who_cut', 'who_assembled']:
                    if source_item.get(role) and not item.get(role):
                        item[role] = source_item[role]
    
    def sync_with_game_state(
        self,
        game: Any,
        agent_ids: List[str],
        counter_positions: List[Tuple[int, int]],
        tick: Optional[int] = None,
        second: Optional[float] = None,
    ):
        """
        Sync tracker state with actual game state to prevent desynchronization.
        
        This reads the real game state and updates internal tracking to match reality:
        - Reads what each agent is actually holding from game.gameObjects[agent_id].item
        - Reads what's on each counter from game.grid.tiles[x][y].item
        - Creates item IDs for any new items discovered
        - Updates agent_holding and counter_items dictionaries
        
        Should be called before processing each batch of actions.
        """
        prev_agent_holding = dict(self.agent_holding)
        prev_counter_items = dict(self.counter_items)

        # Read current game state
        current_agent_items: Dict[str, Optional[str]] = {}
        for agent_id in agent_ids:
            obj = getattr(game, 'gameObjects', {}).get(agent_id)
            current_agent_items[agent_id] = (getattr(obj, 'item', None) or None) if obj is not None else None

        # counter_positions are 1-indexed in logger; tracker keys are 0-indexed
        counter_keys = [(cx - 1, cy - 1) for cx, cy in counter_positions]
        current_counter_items: Dict[Tuple[int, int], Optional[str]] = {}
        for key in counter_keys:
            x, y = key
            tile = game.grid.tiles[x][y]
            current_counter_items[key] = getattr(tile, 'item', None) or None

        # Track which previous IDs are already reused this sync
        used_ids = set()
        new_agent_holding: Dict[str, Optional[str]] = {aid: None for aid in agent_ids}
        new_counter_items: Dict[Tuple[int, int], str] = {}

        def _id_type(item_id: Optional[str]) -> Optional[str]:
            if not item_id or item_id not in self.items:
                return None
            return self.items[item_id]['type']

        # 1) Preserve stable IDs when same holder/location still has same item type
        for agent_id, item_name in current_agent_items.items():
            prev_id = prev_agent_holding.get(agent_id)
            if item_name and prev_id and _id_type(prev_id) == item_name and prev_id not in used_ids:
                new_agent_holding[agent_id] = prev_id
                used_ids.add(prev_id)

        for key, item_name in current_counter_items.items():
            prev_id = prev_counter_items.get(key)
            if item_name and prev_id and _id_type(prev_id) == item_name and prev_id not in used_ids:
                new_counter_items[key] = prev_id
                used_ids.add(prev_id)

        # 2) Detect pickup continuity: counter -> agent
        emptied_counters = [
            key for key, prev_id in prev_counter_items.items()
            if prev_id and key in current_counter_items and not current_counter_items[key]
        ]
        for agent_id, item_name in current_agent_items.items():
            if not item_name or new_agent_holding[agent_id] is not None:
                continue
            matched_id = None
            for key in emptied_counters:
                cand = prev_counter_items.get(key)
                if cand and cand not in used_ids and _id_type(cand) == item_name:
                    matched_id = cand
                    break
            if matched_id:
                new_agent_holding[agent_id] = matched_id
                used_ids.add(matched_id)

        # 3) Detect drop continuity: agent -> counter
        for key, item_name in current_counter_items.items():
            if not item_name or key in new_counter_items:
                continue
            matched_id = None
            for agent_id, prev_id in prev_agent_holding.items():
                if not prev_id or prev_id in used_ids:
                    continue
                if _id_type(prev_id) != item_name:
                    continue
                # agent no longer has this item after step
                now_id = new_agent_holding.get(agent_id)
                now_name = current_agent_items.get(agent_id)
                if now_id != prev_id and now_name != item_name:
                    matched_id = prev_id
                    break
            if matched_id:
                new_counter_items[key] = matched_id
                used_ids.add(matched_id)

        # 4) Reuse any remaining unassigned previous IDs by type
        remaining_prev_ids = [iid for iid in list(prev_agent_holding.values()) + list(prev_counter_items.values()) if iid and iid not in used_ids]

        for agent_id, item_name in current_agent_items.items():
            if not item_name or new_agent_holding[agent_id] is not None:
                continue
            matched_id = None
            for cand in remaining_prev_ids:
                if _id_type(cand) == item_name and cand not in used_ids:
                    matched_id = cand
                    break
            if matched_id:
                new_agent_holding[agent_id] = matched_id
                used_ids.add(matched_id)

        for key, item_name in current_counter_items.items():
            if not item_name or key in new_counter_items:
                continue
            matched_id = None
            for cand in remaining_prev_ids:
                if _id_type(cand) == item_name and cand not in used_ids:
                    matched_id = cand
                    break
            if matched_id:
                new_counter_items[key] = matched_id
                used_ids.add(matched_id)

        # 5) Create IDs for truly new items
        for agent_id, item_name in current_agent_items.items():
            if item_name and new_agent_holding[agent_id] is None:
                new_agent_holding[agent_id] = self.create_item(
                    item_name,
                    agent_id,
                    tick=tick,
                    second=second,
                    source='game_state_sync',
                )

        for key, item_name in current_counter_items.items():
            if item_name and key not in new_counter_items:
                new_counter_items[key] = self.create_item(
                    item_name,
                    '',
                    tick=tick,
                    second=second,
                    source='game_state_sync',
                )

        # Update touches and counters_used based on final mapping
        for agent_id, item_id in new_agent_holding.items():
            if item_id and item_id in self.items:
                if agent_id not in self.items[item_id]['touched_by']:
                    self.items[item_id]['touched_by'].append(agent_id)
                self.items[item_id]['last_touched'] = agent_id

        for key, item_id in new_counter_items.items():
            if item_id and item_id in self.items:
                prev_loc = None
                for old_key, old_id in prev_counter_items.items():
                    if old_id == item_id:
                        prev_loc = old_key
                        break
                if prev_loc != key:
                    self.items[item_id]['counters_used'].append((key, self.items[item_id].get('last_touched', ''), 'drop'))

        self.agent_holding = new_agent_holding
        self.counter_items = new_counter_items
    
    def _get_all_origins(self, item_id: str, visited: Optional[set] = None) -> List[str]:
        """Get all origin IDs recursively."""
        if item_id not in self.items:
            return []
        if visited is None:
            visited = set()
        if item_id in visited:
            return []
        visited.add(item_id)
        origins = []
        for origin_id in self.items[item_id]['origins'].values():
            if not origin_id or origin_id in visited:
                continue
            origins.append(origin_id)
            origins.extend(self._get_all_origins(origin_id, visited))
        return origins
    
    def _empty_data(self) -> Dict:
        """Return empty item data structure."""
        return {
            'item_id': '', 'item_type': '', 'creation_index': '', 'created_by': '',
            'created_tick': '', 'created_second': '', 'created_source': '',
            'tomato_id': '', 'plate_id': '', 'tomato_cut_id': '', 'tomato_salad_id': '', 'tomato_delivered_id': '',
            'last_touched': '', 'touched_list': '',
            'is_item_collaboration': False, 'is_exchange_collaboration': False,
            'who_picked_tomato': '', 'who_picked_plate': '', 'who_cutted': '',
            'who_assembled': '', 'who_delivered': '', 'number_of_counters_used': 0
        }
