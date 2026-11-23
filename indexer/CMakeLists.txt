cmake_minimum_required(VERSION 3.16)
project(meta_indexer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 압축 푼 ONNX Runtime 폴더 / hnswlib 폴더 (기본: 저장소 루트에 둔 것)
set(ORT_DIR "${CMAKE_SOURCE_DIR}/../onnxruntime" CACHE PATH "ONNX Runtime 경로")
set(HNSW_DIR "${CMAKE_SOURCE_DIR}/../hnswlib" CACHE PATH "hnswlib 경로")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)

add_executable(indexer
  src/indexer_main.cpp
  src/embedder.cpp
  src/summary.cpp
  src/stitcher.cpp
  src/metadata_parser.cpp)

target_include_directories(indexer PRIVATE src ${ORT_DIR}/include ${HNSW_DIR})
target_link_libraries(indexer PRIVATE ${OpenCV_LIBS} ${ORT_DIR}/lib/libonnxruntime.so)
# 실행 시 libonnxruntime.so를 찾도록
set_target_properties(indexer PROPERTIES BUILD_RPATH "${ORT_DIR}/lib")

# 검색 서버: 자연어 → CLIP 텍스트 인코더(C++ BPE 토크나이저) → hnswlib → HTTP(JSON·이미지)
#   실시간 모드(--live): 메타데이터를 따라 읽으며 잇기·임베딩해 인덱스에 바로 추가 (live_indexer + 인덱서와 같은 부품)
find_package(Threads REQUIRED)
add_executable(search_server
  src/search_main.cpp
  src/clip_tokenizer.cpp
  src/text_embedder.cpp
  src/http_server.cpp
  src/http_client.cpp
  src/summary.cpp
  src/metadata_parser.cpp
  src/live_indexer.cpp
  src/embedder.cpp
  src/stitcher.cpp)
target_include_directories(search_server PRIVATE src ${ORT_DIR}/include ${HNSW_DIR} ${CMAKE_SOURCE_DIR}/../third_party)
target_link_libraries(search_server PRIVATE ${OpenCV_LIBS} ${ORT_DIR}/lib/libonnxruntime.so Threads::Threads)
set_target_properties(search_server PROPERTIES BUILD_RPATH "${ORT_DIR}/lib")

# 챗봇 서버: 말 한 문장 → 규칙(온톨로지·정규식) → 모르면 로컬 LLM(Ollama, JSON 스키마로 선택지 고정) → 도구 하나 실행
#   도구: 사람 검색(search_server /search) · 카메라 탐색(ONVIF WS-Discovery) · 카메라 제어(SUNAPI) · 등록
#   OpenCV·ONNX Runtime 없이 빌드된다
add_executable(chat_server
  src/chat_main.cpp
  src/chat_rules.cpp
  src/wsdiscovery.cpp
  src/sunapi.cpp
  src/http_server.cpp
  src/http_client.cpp)
target_include_directories(chat_server PRIVATE src ${CMAKE_SOURCE_DIR}/../third_party)
target_link_libraries(chat_server PRIVATE Threads::Threads)
