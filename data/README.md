# Data

Put the files for your study in this folder.

To prepare a study:

1. **Graph:** build a follower or reply network from one of the sources listed in the [main README](../README.md#data-sources). Sample it down to a manageable number of agents.
2. **Communities:** label each agent with a community from a standard community detection method, such as Louvain.
3. **Personas:** give each agent a short persona, such as a few of its own public posts.
4. **Messages:** choose the posts to spread, with a few well-followed seed accounts for each.
5. **Answers:** call the chat API live, or record each model's share probability once per agent, message format and sharer count, then replay the study from those tables.

The file formats the command line expects are described in the [main README](../README.md#input-format-for-logged-answers).
