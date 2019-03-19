# Developers

## Coding Guide Lines
- [Python](https://www.python.org/dev/peps/pep-0008/)
- Java: 
- Kotlin:
- Scala:

## Documentation
- [Sphinx quickstart](http://www.sphinx-doc.org/en/master/usage/quickstart.html)
- [Sample Doc](https://matplotlib.org/sampledoc/index.html)
- Sphinx Cheatsheet
    - [1](http://openalea.gforge.inria.fr/doc/openalea/doc/_build/html/source/sphinx/rest_syntax.html)
- [Example](http://www.sphinx-doc.org/en/master/examples.html)

## Build Tool
- [Gradle](https://gradle.org/)

## Continuous Integration
- [GIT Travis](https://travis-ci.org/)

## IntelliJ Integration


## Python Notebook Server
```
ipython notebook --no-browser --port=8889
ssh -N -f -L localhost:8888:localhost:8889 remote_user@remote_host
```

## GIT Commands
**Sub modules notes:**

```commandline
#add submodule and define the master branch as the one you want to track  
git submodule add -b master [URL to Git repo]     
git submodule init

#update your submodule --remote fetches new commits in the submodules 
# and updates the working tree to the commit described by the branch  
# pull all changes for the submodules
git submodule update --remote
 ---or---
# pull all changes in the repo including changes in the submodules
git pull --recurse-submodules


# update submodule in the master branch
# skip this if you use --recurse-submodules
# and have the master branch checked out
cd [submodule directory]
git checkout master
git pull

# commit the change in main repo
# to use the latest commit in master of the submodule
cd ..
git add [submodule directory]
git commit -m "move submodule to latest commit in master"

# share your changes
git push
```

### Note:
- Good practice to run below command to get rid of python compiled 
files to avoid errors in new environment:

```
 find . -name "*.pyc" -exec rm -f {} \;
 ```

