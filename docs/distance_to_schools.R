# Location of local schools with geocodes
ames_schools <- AmesHousing::ames_schools

# Function to compute distance to 
distance_to_schools <- function(lon, lat) {
  x <- c(lon, ames_schools$Longitude)
  y <- c(lat, ames_schools$Latitude)
  res <- as.matrix(dist(cbind(x, y)))[-1L, 1L, drop = TRUE]
  names(res) <- ames_schools$School
  res
}

# Add column indicating nearest school (categorical)
res <- NULL
for (i in seq_len(nrow(ames))) {
  res <- rbind(res, distance_to_schools(ames$Longitude[i], ames$Latitude[i]))
}
nearest_school <- ames_schools$School[apply(res, MARGIN = 1, FUN = which.min)]
ames$nearest_school <- as.factor(nearest_school)

set.seed(159)
rfo <- ranger(Sale_Price ~ ., data = ames, importance = "impurity")
vip::vip(rfo, num_features = 20)
