#https://cran.r-project.org/web/packages/mallet/vignettes/mallet.html

system("arch -arm64 brew install openjdk") 
system("sudo ln -sfn /opt/homebrew/opt/openjdk/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk.jdk") 
system("export CPPFLAGS='-I/opt/homebrew/opt/openjdk/include'")


# In R console
install.packages("rJava")
.rs.restartR() 
library(rJava)
install.packages("mallet")
