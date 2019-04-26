state = State((train[0][1],train[0][2]), kg)
update_node = state.find_all_neighbors()[0]
print(update_node)
print(state.subgraphs)
state.update(update_node)
print(state.subgraphs)
update_node = state.find_all_neighbors()[18]
print(update_node)
state.update(update_node)
print(state.subgraphs)