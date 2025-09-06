cmake_minimum_required(VERSION 3.5)
project(robotis_omy)

#── 1) ament, ROS2, Eigen, PkgConfig ───────────────────────────────
find_package(ament_cmake    REQUIRED)
find_package(rclcpp         REQUIRED)
find_package(Eigen3         REQUIRED)
find_package(PkgConfig      REQUIRED)

#── 2) pkg-config 으로 MuJoCo 찾기 ─────────────────────────────────
pkg_check_modules(MUJOCO REQUIRED mujoco)

#── 3) 실행 파일 추가 ───────────────────────────────────────────────
add_executable(unified_admittance_trajectory_control
  src/unified_admittance_trajectory_control.cpp
)

#── 4) include / link 경로 ───────────────────────────────────────────
#    헤더 디렉토리는 타겟에 PUBLIC(또는 PRIVATE)으로 지정
target_include_directories(unified_admittance_trajectory_control
  PUBLIC
    ${MUJOCO_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

#    라이브러리 디렉토리는 전역 등록해도 되고, 타겟에 바로 링크해도 됩니다
link_directories(${MUJOCO_LIBRARY_DIRS})

#── 5) ROS2·Eigen 의존성 링크 ───────────────────────────────────────
ament_target_dependencies(unified_admittance_trajectory_control
  rclcpp
  Eigen3
)

#── 6) MuJoCo 라이브러리 링크 ───────────────────────────────────────
target_link_libraries(unified_admittance_trajectory_control
  ${MUJOCO_LIBRARIES}   # 예: -lmujoco
)

#── 7) 설치 규칙 ───────────────────────────────────────────────────
install(TARGETS
  unified_admittance_trajectory_control
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()

