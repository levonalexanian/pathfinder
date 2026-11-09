from setuptools import setup

package_name = "pathfinder_rerun"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Levon Alexanian",
    maintainer_email="levon.alexanian10@gmail.com",
    description="Bridges PathFinder ROS topics into the Rerun viewer.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "rerun_bridge = pathfinder_rerun.bridge_node:main",
        ],
    },
)
