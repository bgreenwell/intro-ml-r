# Load required packages
library(plotly)

# Load the email spam data
data(spam, package = "kernlab")

# 3D scatter plot
p <- plot_ly(spam, 
             x = ~log(charExclamation), 
             y = ~log(capitalLong), 
             z = ~log(hp), 
             color = ~type, 
             colors = c('#BF382A', '#0C4B8E')) %>%
  add_markers(opacity = 0.5) %>%
  layout(scene = list(xaxis = list(title = "charExclamation"),
                      yaxis = list(title = "capitalLong"),
                      zaxis = list(title = "hp")))
