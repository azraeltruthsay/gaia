Dev Journal - January 31, 2026                                                                                                                                                                                                            
                                                                                                                                                                                                                                            
  Feature Proposal: Dynamic Status, Resource-Awareness, and Autonomous Knowledge Acquisition                                                                                                                                                
                                                                                                                                                                                                                                            
  1. Summary                                                                                                                                                                                                                                
                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                            
  This document consolidates two parallel streams of development and feature proposals:                                                                                                                                                     
   1. Dynamic Status & "Distracted" State: A proposal for GAIA to have a resource-aware "distracted" state and to communicate her internal status to the user via dynamic flags.                                                            
   2. Autonomous Knowledge Acquisition: An implementation of query-time learning, allowing GAIA to discover, embed, and retrieve knowledge on-the-fly.                                                                                      
                                                                                                                                                                                                                                            
  This merged document provides a holistic view of creating a more transparent, resilient, and intelligent GAIA.                                                                                                                            
                                                                                                                                                                                                                                            
  2. Core Concepts                                                                                                                                                                                                                          
                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                            
  2.1. The "Distracted" State and Resource-Aware Model Loading                                                                                                                                                                              
                                                                                                                                                                                                                                            
  The "Distracted" state is a direct response to the physical constraints of the system GAIA is running on.                                                                                                                                 
                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                            
   * Trigger: This state would be triggered when a monitoring component detects that GPU utilization has passed a predefined threshold (e.g., >80% for more than 5 seconds). This indicates that the GPU is busy with other tasks, such as  
     training, running another model, or a heavy workload from another user.                                                                                                                                                                
   * Behavior:                                                                                                                                                                                                                              
       * When in a "Distracted" state, the ModelPool will not attempt to load or use the gpu_prime model.                                                                                                                                   
       * All cognitive tasks, including response generation and stream observation, will fall back to the lite CPU model.                                                                                                                   
       * This will result in a noticeable change in GAIA's behavior: responses will be generated faster but may be less nuanced or detailed. The back-and-forth of the lite model observing itself could slow down the final response time  
         and reduce its overall effectiveness.                                                                                                                                                                                              
   * Purpose: This mechanism makes GAIA more resilient and responsive. Instead of waiting for a busy GPU and potentially timing out, she can provide a "best-effort" response using available resources. It also provides a naturalistic    
     explanation for variations in her performance.
                                                                                                                                                                                                                                               
  2.2. Dynamic Status Flags for Transparency                                                                                                                                                                                                
                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                            
  To make GAIA's internal state transparent to the user, we will introduce a system of dynamic status flags. These flags will be displayed alongside her responses in supported UIs (like Discord).                                         
                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                            
   * Proposed Flags:                                                                                                                                                                                                                        
       * `#Status`: Indicates GAIA's current operational state.
           * #Ready: GPU is available, and Prime model is ready.
           * #Distracted: GPU is busy; operating in Lite mode.
           * #Studying: A training process is active.
           * #Observing: Passively watching a channel without being the primary respondent.
       * `#Model`: The model that generated the response.
           * #Prime: The primary GPU model.
           * #Lite: The CPU-based model.
           * #Oracle: An external API (e.g., GPT-4, Gemini).
           * #Dev: A development or test model.
       * `#Observer`: The model that is observing the response stream. (e.g., #Observer:Lite)
       * `#Confidence`: The model's confidence in its response, on a scale (e.g., #Confidence:85%).
       * `#Intent`: The system's interpretation of the user's request (e.g., #Intent:question, #Intent:command).


   * Example Discord Response:
      > GAIA #Ready #Prime #Confidence:95% #Intent:question
      >
      > Heimr is a fascinating campaign setting. From my knowledge...


      
      
      2.3. The "Hydrated" Cognition Packet


  A key concern is that adding all this metadata to the GCP will "pollute" it with tokens that are not useful for the model's reasoning process, thus reducing efficiency. The proposed solution is to treat the GCP as a "hydrated" data
  structure.


   * Dual Representation: The GCP will contain two representations of the status information:
       1. Machine-Readable (Encoded): A highly compact, semantically encoded representation for the model to use. For example, #Status:Distracted could be encoded as S:D. This is what the model would see in its prompt context.
       2. Human-Readable (Hydrated): The full, human-friendly string (e.g., #Status:Distracted), which is used by the UI to display the flags. This data would be passed along in the GCP's metadata but not necessarily included in the
          prompt itself.

   * Semantic Encoding Cheat Sheet: To ensure the model can efficiently interpret the encoded flags, we will provide it with a "cheat sheet" in its system prompt.

  3. Autonomous Knowledge Acquisition (Claude's Implementation Notes)

  This section details the implementation of query-time learning, which is the first step towards a self-evolving GAIA.


   * Core Workflow:
       1. A user's query is received.
       2. The RAG (Retrieval Augmented Generation) system queries the vector store.
       3. If no results are found, the _knowledge_acquisition_workflow is triggered.
       4. This workflow calls the find_relevant_documents MCP tool to search the filesystem for documents matching the query.
       5. If documents are found, the embed_documents MCP tool is called.
       6. The embed_documents tool hashes the document and checks against the KnowledgeIndex to avoid re-embedding unchanged files.
       7. If the document is new or changed, it is chunked and embedded, and the new hash is stored.
       8. The RAG query is re-run with the newly embedded knowledge.
       9. The results are used to generate a response.


   * Key Files Modified by Claude:
       * app/mcp_lite_server.py: To fix path resolution and add hashing and duplicate detection to the embedding flow.
       * app/utils/vector_indexer.py: To add text chunking.
       * app/cognition/agent_core.py: To add logging and on-demand loading for the observer model.
       * docker-compose.single.yml: To fix malformed YAML.
       * app/gaia_constants.json: To enable the observer model.

  4. Next Steps


   * Discord Integration: Migrate the Discord connector from the gaia-assistant monolith to the new SOA architecture.
   * Implement Status Flags: Implement the dynamic status flags and the "Distracted" state.
   * Continue RAG Debugging: The RAG system is still not working as expected. Further debugging is needed.
  """)

      
      
      
      
      
      
      
     
     
     
