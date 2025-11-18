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
