cmake_minimum_required(VERSION 2.8)
project(srrg_hbst)

#ds INSTALL RULE
install(DIRECTORY "srrg_hbst" DESTINATION include)

#ds UNINSTALL RULE - FOR NOW ONLY SUPPORTED ON LINUX AND NOT THE CLEAN WAY - use at own risk!
add_custom_target(uninstall COMMAND
echo "removing HBST header directory: /usr/local/include/srrg_hbst" &&
rm -r -v /usr/local/include/srrg_hbst &&
echo "removing HBST pkgconfig file: /usr/local/lib/pkgconfig/srrg_hbst.pc" &&
rm -v /usr/local/lib/pkgconfig/srrg_hbst.pc &&
echo "removing HBST CMake directory: /usr/local/share/srrg_hbst" &&
rm -r -v /usr/local/share/srrg_hbst)

#ds check if catkin is available on the building system
find_package(catkin QUIET)
if(catkin_FOUND)
  #ds define header-only catkin package
  message("${PROJECT_NAME}|found catkin, building unittests")
  catkin_package(INCLUDE_DIRS .)
  
  #ds source include directories for unittest build
  include_directories(SYSTEM ${catkin_INCLUDE_DIRS})
  include_directories(${PROJECT_SOURCE_DIR})
  
  #ds HBST generic compilation flags - ADD THEM IN YOUR PROJECT AS WELL TO ENABLE THEM (HBST is header-only)
  # - SRRG_MERGE_DESCRIPTORS: HBST checks for identical descriptors stemming from multiple images and represents them with a single entity
  #two descriptors are merged if their distance is within a certain threshold
  #the threshold can be controlled with the static variable maximum_distance_for_merge
  #a value of maximum_distance_for_merge=0 means they have to be identical
  #the higher this value, the more compact and efficient becomes the tree at the cost of matching accuracy
  #this flag is recommended for large-scale datasets and lifelong recognition applications
  
  #ds unittest targets without merging
  catkin_add_gtest(test_search tests/test_search.cpp)
  catkin_add_gtest(test_streaming tests/test_streaming.cpp)
  target_link_libraries(test_streaming ${catkin_LIBRARIES})
  
  #ds unittest targets with merging - this should not change the behavior - configure default flags and build
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Werror -pedantic")
  catkin_add_gtest(test_search_merging tests/test_search.cpp)
  catkin_add_gtest(test_streaming_merging tests/test_streaming.cpp)
  target_compile_definitions(test_streaming_merging PRIVATE SRRG_MERGE_DESCRIPTORS)
  target_link_libraries(test_streaming_merging ${catkin_LIBRARIES})
endif()
