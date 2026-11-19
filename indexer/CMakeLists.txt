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
find_package(Threads REQUIRED)
add_executable(search_server
  src/search_main.cpp
  src/clip_tokenizer.cpp
  src/text_embedder.cpp
  src/http_server.cpp
  src/summary.cpp
  src/metadata_parser.cpp)
target_include_directories(search_server PRIVATE src ${ORT_DIR}/include ${HNSW_DIR} ${CMAKE_SOURCE_DIR}/../third_party)
target_link_libraries(search_server PRIVATE ${OpenCV_LIBS} ${ORT_DIR}/lib/libonnxruntime.so Threads::Threads)
set_target_properties(search_server PROPERTIES BUILD_RPATH "${ORT_DIR}/lib")
