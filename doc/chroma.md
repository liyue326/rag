sqlite3 your_file.bin
.tables           # 列出所有表
sqlite> .tables
  collection_metadata                embeddings                       
  collections                        embeddings_queue                 
  databases                          embeddings_queue_config          
  embedding_fulltext_search          maintenance_log                  
  embedding_fulltext_search_config   max_seq_id                       
  embedding_fulltext_search_content  migrations                       
  embedding_fulltext_search_data     segment_metadata                 
  embedding_fulltext_search_docsize  segments                         
  embedding_fulltext_search_idx      tenants                          
  embedding_metadata     
SELECT * FROM embeddings LIMIT 5;  # 查看前5条数据
sqlite> SELECT * FROM embeddings LIMIT 5;
  1|4188f1aa-0e64-484c-8288-d3ff8f9585ac|ad68aa17-7d96-4588-ba8b-5a7a3bdd1526||2025-04-01 12:50:54
  2|4188f1aa-0e64-484c-8288-d3ff8f9585ac|ae2c2874-a80d-4dad-b118-d7612bd768bb||2025-04-01 12:52:22
  3|4188f1aa-0e64-484c-8288-d3ff8f9585ac|0396b155-0959-4297-9997-394c4bf1db94||2025-04-01 12:56:05
  4|4188f1aa-0e64-484c-8288-d3ff8f9585ac|6df4013b-9cc9-49c7-85ee-9ec9b18244d4||2025-04-01 13:01:18
  5|4188f1aa-0e64-484c-8288-d3ff8f9585ac|94cf318b-5fbf-4fd8-bfea-b47a59640e0f||2025-04-01 13:04:35