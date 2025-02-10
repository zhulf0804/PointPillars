pip uninstall pointpillars -y
rm -rf build
rm -rf pointpillars.egg-info
pip install .
cd ../..
python -c "from pointpillars.model import PillarLayer"