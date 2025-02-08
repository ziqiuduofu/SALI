library(caret)
library(missForest) 
library(doParallel)
cl <- makePSOCKcluster(24)
registerDoParallel(cl)
library(tidymodels)
library(bonsai)
library(stacks)
source("/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/机器学习r/tidyfuncs4cls2.R")

# 读取数据
# file.choose()
Heart <- readr::read_csv(file.choose())
colnames(Heart) 
# 修正变量类型
# 将分类变量转换为factor
for(i in c(1:7)){ 
  Heart[[i]] <- factor(Heart[[i]])
}

# 删除无关变量在此处进行
Heart$Id <- NULL
# 删除含有缺失值的样本在此处进行，填充缺失值在后面
Heart <- na.omit(Heart)
# Heart <- Heart %>%
#   drop_na(Thal)

# 变量类型修正后数据概况
skimr::skim(Heart)    

library(dplyr)

# 使用dplyr替换ending列的值
training_set <- training_set %>%
  mutate(ending = ifelse(ending == 1, "No", ifelse(ending == 2, "Yes", ending)))
validation_set <- validation_set %>%
  mutate(ending = ifelse(ending == 1, "No", ifelse(ending == 2, "Yes", ending)))
validation_set2 <- validation_set2 %>%
  mutate(ending = ifelse(ending == 2, "No", ifelse(ending == 1, "Yes", ending)))

validation_set2 <- validation_set2 %>%
  mutate(rrt = ifelse(rrt == 3, "0", ifelse(rrt == 4, "1", rrt)))

# 查看处理后的数据集
head(training_set)

str(validation_set2)
# 设定阳性类别和阴性类别
yourpositivelevel <- "Yes"
yournegativelevel <- "No"
# 转换因变量的因子水平，将阳性类别设定为第二个水平
levels(validation_set2$ending)
table(validation_set2$ending)
training_set$ending <- factor(
  training_set$ending,
  levels = c(yournegativelevel, yourpositivelevel)
)
validation_set$ventilation_status <- factor(validation_set$ventilation_status)
validation_set$ending <- factor(
  validation_set$ending,
  levels = c(yournegativelevel, yourpositivelevel)
)
validation_set2$ending <- factor(
  validation_set2$ending,
  levels = c(yournegativelevel, yourpositivelevel)
)
levels(Heart$AJCC.M)
table(Heart$AJCC.M)
# 数据拆分
set.seed(2024)
datasplit <- initial_split(Heart, prop = 0.7, strata = AJCC.M)
traindata <- training(datasplit)
testdata <- testing(datasplit)





#不拆分，外部验证
traindata <- Heart
aa <- readr::read_csv(file.choose())
colnames(aa) 
# 修正变量类型
# 将分类变量转换为factor
for(i in c(1:9)){ 
  aa[[i]] <- factor(aa[[i]])
}
testdata <- aa

# 重抽样设定-10折交叉验证
set.seed(42)
folds <- vfold_cv(training_set, v = 10, strata = ending)
folds

# 数据预处理配方
datarecipe_dt <- recipe(formula = ending ~ ., training_set)
datarecipe_dt

datarecipe_rf <- recipe(formula = ending ~ ., training_set)
datarecipe_rf

datarecipe_xgboost <- recipe(formula = ending ~ ., training_set) %>%
  step_dummy(all_nominal_predictors(), 
             naming = new_dummy_names)
datarecipe_xgboost

datarecipe_enet <- recipe(formula = ending ~ ., training_set) %>%
  step_dummy(all_nominal_predictors(), 
             naming = new_dummy_names) %>% 
  step_normalize(all_predictors())
datarecipe_enet



datarecipe_svm <- recipe(formula = ending ~ ., training_set) %>%
  step_dummy(all_nominal_predictors(), 
             naming = new_dummy_names) %>% 
  step_normalize(all_predictors())
datarecipe_svm



datarecipe_mlp <- recipe(formula = ending ~ ., training_set) %>%
  step_dummy(all_nominal_predictors(), 
             naming = new_dummy_names) %>% 
  step_range(all_predictors())
datarecipe_mlp


datarecipe_lightgbm <- recipe(formula = ending ~ ., training_set) %>%
  step_dummy(all_nominal_predictors(), 
             naming = new_dummy_names)
datarecipe_lightgbm


datarecipe_knn <- recipe(formula = ending ~ ., training_set) %>%
  step_dummy(all_nominal_predictors(), 
             naming = new_dummy_names) %>% 
  step_normalize(all_predictors())
datarecipe_knn


datarecipe_logistic <- recipe(formula = ending ~ ., training_set)
datarecipe_logistic





# 设定模型
model_dt <- decision_tree(
  mode = "classification",
  engine = "rpart",
  tree_depth = tune(),
  min_n = tune(),
  cost_complexity = tune()
) %>%
  set_args(model=TRUE)
model_dt

# workflow
wk_dt <- 
  workflow() %>%
  add_recipe(datarecipe_dt) %>%
  add_model(model_dt)
wk_dt

# 设定模型
model_rf <- rand_forest(
  mode = "classification",
  engine = "randomForest", # ranger
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_args(importance = T)
model_rf

# workflow
wk_rf <- 
  workflow() %>%
  add_recipe(datarecipe_rf) %>%
  add_model(model_rf)
wk_rf



# 设定模型
model_xgboost <- boost_tree(
  mode = "classification",
  engine = "xgboost",
  mtry = tune(),
  trees = 1000,
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  stop_iter = 25
) %>%
  set_args(validation = 0.2,
           event_level = "second")
model_xgboost

# workflow
wk_xgboost <- 
  workflow() %>%
  add_recipe(datarecipe_xgboost) %>%
  add_model(model_xgboost)
wk_xgboost




# 设定模型
model_enet <- logistic_reg(
  mode = "classification",
  engine = "glmnet",
  # mixture = 1,   # LASSO
  # mixture = 0,  # 岭回归
  mixture = tune(),
  penalty = tune()
)
model_enet

# workflow
wk_enet <- 
  workflow() %>%
  add_recipe(datarecipe_enet) %>%
  add_model(model_enet)
wk_enet



# 设定模型
model_svm <- svm_rbf(
  mode = "classification",
  engine = "kernlab",
  cost = tune(),
  rbf_sigma = tune()
)
model_svm

# workflow
wk_svm <- 
  workflow() %>%
  add_recipe(datarecipe_svm) %>%
  add_model(model_svm)
wk_svm





# 设定模型
model_mlp <- mlp(
  mode = "classification",
  engine = "nnet",
  hidden_units = tune(),
  penalty = tune(),
  epochs = tune()
) 
model_mlp

# workflow
wk_mlp <- 
  workflow() %>%
  add_recipe(datarecipe_mlp) %>%
  add_model(model_mlp)
wk_mlp




# 设定模型
model_lightgbm <- boost_tree(
  mode = "classification",
  engine = "lightgbm",
  tree_depth = tune(),
  trees = tune(),
  learn_rate = tune(),
  mtry = tune(),
  min_n = tune(),
  loss_reduction = tune()
)
model_lightgbm

# workflow
wk_lightgbm <- 
  workflow() %>%
  add_recipe(datarecipe_lightgbm) %>%
  add_model(model_lightgbm)
wk_lightgbm




# 设定模型
model_knn <- nearest_neighbor(
  mode = "classification",
  engine = "kknn",
  
  neighbors = tune(),
  weight_func = tune(),
  dist_power = 2
)
model_knn

# workflow
wk_knn <- 
  workflow() %>%
  add_recipe(datarecipe_knn) %>%
  add_model(model_knn)
wk_knn




# 设定模型
model_logistic <- logistic_reg(
  mode = "classification",
  engine = "glm"
)
model_logistic

# workflow
wk_logistic <- 
  workflow() %>%
  add_recipe(datarecipe_logistic) %>%
  add_model(model_logistic)
wk_logistic




#########################  超参数寻优贝叶斯优化

# 超参数寻优网格
set.seed(42)
hpgrid_dt <- parameters(
  tree_depth(range = c(3, 7)),
  min_n(range = c(5, 10)),
  cost_complexity(range = c(-6, -3))
) %>%
  # grid_regular(levels = c(3, 2, 4)) # 常规网格
  grid_random(size = 20) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_dt
log10(hpgrid_dt$cost_complexity)
# 网格也可以自己手动生成expand.grid()
# hpgrid_dt <- expand.grid(
#   tree_depth = c(2:5),
#   min_n = c(5, 11),
#   cost_complexity = 10^(-5:-1)
# )

# 交叉验证网格搜索过程
set.seed(42)
tune_dt <- wk_dt %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_dt,
    metrics = metric_set(yardstick::accuracy, 
                         yardstick::roc_auc, 
                         yardstick::pr_auc),
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )
# 贝叶斯优化超参数
set.seed(42)
tune_dt <- wk_dt %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    metrics = metric_set(yardstick::roc_auc,
                         yardstick::accuracy, 
                         yardstick::pr_auc),
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )





# 超参数寻优网格
set.seed(42)
hpgrid_rf <- parameters(
  mtry(range = c(2, 10)), 
  trees(range = c(200, 500)),
  min_n(range = c(20, 50))
) %>%
  # grid_regular(levels = c(3, 2, 4)) # 常规网格
  grid_random(size = 20) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_rf
# 网格也可以自己手动生成expand.grid()

# 交叉验证网格搜索过程
set.seed(42)
tune_rf <- wk_rf %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_rf,
    metrics = metric_set(yardstick::accuracy, 
                         yardstick::roc_auc, 
                         yardstick::pr_auc),
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )


param_rf <- model_rf %>%
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(2, 10)))

# 贝叶斯优化超参数
set.seed(42)
tune_rf <- wk_rf %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    param_info = param_rf,
    metrics = metric_set(yardstick::roc_auc,
                         yardstick::accuracy, 
                         yardstick::pr_auc),
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )





# 超参数寻优网格
set.seed(42)
hpgrid_xgboost <- parameters(
  mtry(range = c(2, 8)),
  min_n(range = c(5, 20)),
  tree_depth(range = c(1, 3)),
  learn_rate(range = c(-3, -1)),
  loss_reduction(range = c(-3, 0)),
  sample_prop(range = c(0.8, 1))
) %>%
  # grid_regular(levels = c(3, 2, 2, 3, 2, 2)) # 常规网格
  grid_random(size = 20) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_xgboost
# 网格也可以自己手动生成expand.grid()

# 交叉验证网格搜索过程
set.seed(42)
tune_xgboost <- wk_xgboost %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_xgboost,
    metrics = metric_set(yardstick::accuracy, 
                         yardstick::roc_auc, 
                         yardstick::pr_auc),
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )

# 更新超参数范围
param_xgboost <- model_xgboost %>%
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(2, 10)))

# 贝叶斯优化超参数
set.seed(42)
tune_xgboost <- wk_xgboost %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    param_info = param_xgboost,
    metrics = metric_set(yardstick::roc_auc,
                         yardstick::accuracy, 
                         yardstick::pr_auc),
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )




# 超参数寻优网格
set.seed(42)
hpgrid_enet <- parameters(
  mixture(),
  penalty(range = c(-5, 0))
) %>%
  grid_regular(levels = c(5, 20)) # 常规网格
# grid_random(size = 5) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_enet
# 网格也可以自己手动生成expand.grid()

# 交叉验证网格搜索过程
set.seed(42)
tune_enet <- wk_enet %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_enet,
    metrics = metric_set(yardstick::accuracy, 
                         yardstick::roc_auc, 
                         yardstick::pr_auc),
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )


set.seed(42)
tune_enet <- wk_enet %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    metrics = metric_set(yardstick::roc_auc,
                         yardstick::accuracy, 
                         yardstick::pr_auc),
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )





# 超参数寻优网格
set.seed(42)
hpgrid_svm <- parameters(
  cost(range = c(-5, 5)), 
  rbf_sigma(range = c(-4, -1))
) %>%
  # grid_regular(levels = c(2,3)) # 常规网格
  grid_random(size = 20) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_svm
# 网格也可以自己手动生成expand.grid()

# 交叉验证网格搜索过程
set.seed(42)
tune_svm <- wk_svm %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_svm,
    metrics = metric_set(yardstick::accuracy, 
                         yardstick::roc_auc, 
                         yardstick::pr_auc),
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )

set.seed(42)
tune_svm <- wk_svm %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    metrics = metric_set(yardstick::roc_auc, 
                         yardstick::accuracy, 
                         yardstick::pr_auc),
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )






# 超参数寻优网格
set.seed(42)
hpgrid_mlp <- parameters(
  hidden_units(range = c(15, 24)),
  penalty(range = c(-3, 0)),
  epochs(range = c(50, 150))
) %>%
  grid_regular(levels = 3) # 常规网格
# grid_random(size = 5) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_mlp
# 网格也可以自己手动生成expand.grid()

# 交叉验证网格搜索过程
set.seed(42)
tune_mlp <- wk_mlp %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_mlp,
    metrics = metric_set(yardstick::accuracy, 
                         yardstick::roc_auc, 
                         yardstick::pr_auc),
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )


# 贝叶斯优化超参数
set.seed(42)
tune_mlp <- wk_mlp %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    metrics = metric_set(yardstick::roc_auc, 
                         yardstick::accuracy, 
                         yardstick::pr_auc),
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )




# 超参数寻优网格
set.seed(42)
hpgrid_lightgbm <- parameters(
  tree_depth(range = c(1, 3)),
  trees(range = c(100, 500)),
  learn_rate(range = c(-3, -1)),
  mtry(range = c(2, 8)),
  min_n(range = c(5, 10)),
  loss_reduction(range = c(-3, 0))
) %>%
  # grid_regular(levels = c(3, 2, 2, 3, 2, 2)) # 常规网格
  grid_random(size = 20) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_lightgbm
# 网格也可以自己手动生成expand.grid()

# 交叉验证网格搜索过程
set.seed(42)
tune_lightgbm <- wk_lightgbm %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_lightgbm,
    metrics = metric_set(yardstick::accuracy, 
                         yardstick::roc_auc, 
                         yardstick::pr_auc),
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )

# 更新超参数范围
param_lightgbm <- model_lightgbm %>%
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(2, 10)))

# 贝叶斯优化超参数
set.seed(42)
tune_lightgbm <- wk_lightgbm %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    param_info = param_lightgbm,
    metrics = metric_set(yardstick::roc_auc,
                         yardstick::accuracy, 
                         yardstick::pr_auc),
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )




# 超参数寻优网格
set.seed(42)
hpgrid_knn <- parameters(
  neighbors(range = c(3, 11)),
  weight_func()
) %>%
  # grid_regular(levels = c(5)) # 常规网格
  grid_random(size = 20) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_knn
# 网格也可以自己手动生成expand.grid()
# 交叉验证网格搜索过程
set.seed(42)
tune_knn <- wk_knn %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_knn,
    metrics = metric_set(yardstick::accuracy, 
                         yardstick::roc_auc, 
                         yardstick::pr_auc),
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )


# 贝叶斯优化超参数
set.seed(42)
tune_knn <- wk_knn %>%
  tune_bayes(
    resamples = folds,
    initial = 20,
    iter = 50,
    metrics = metric_set(yardstick::roc_auc,
                         yardstick::accuracy, 
                         yardstick::pr_auc),
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )





########################  超参数寻优结束


# 交叉验证结果
eval_tune_dt <- tune_dt %>%
  collect_metrics()
eval_tune_dt

# 图示
# autoplot(tune_dt)
tune_dt_plot <- eval_tune_dt %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'cost_complexity', values = ~cost_complexity),
      list(label = 'tree_depth', values = ~tree_depth),
      list(label = 'min_n', values = ~min_n)
    )
  ) %>%
  plotly::layout(title = "DT HPO Guided by AUCROC")
tune_dt_plot
scope <- plotly::kaleido()
scope$transform(tune_dt_plot, "tune_dt_plot.pdf")

# 经过交叉验证得到的最优超参数
hpbest_dt <- tune_dt %>%
  select_by_one_std_err(metric = "roc_auc", desc(cost_complexity))
hpbest_dt







# 交叉验证结果
eval_tune_rf <- tune_rf %>%
  collect_metrics()
eval_tune_rf

# 图示
# autoplot(tune_rf)
eval_tune_rf %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'mtry', values = ~mtry),
      list(label = 'trees', values = ~trees),
      list(label = 'min_n', values = ~min_n)
    )
  ) %>%
  plotly::layout(title = "RF HPO Guided by AUCROC")


# autoplot(tune_rf)
tune_rf_plot <- eval_tune_rf %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'mtry', values = ~mtry),
      list(label = 'trees', values = ~trees),
      list(label = 'min_n', values = ~min_n)
    )
  ) %>%
  plotly::layout(title = "RF HPO Guided by AUCROC")
tune_rf_plot
scope <- plotly::kaleido()
scope$transform(tune_rf_plot, "tune_rf_plot.pdf")


# 经过交叉验证得到的最优超参数
hpbest_rf <- tune_rf %>%
  select_by_one_std_err(metric = "roc_auc", desc(min_n))
hpbest_rf





# 交叉验证结果
eval_tune_xgboost <- tune_xgboost %>%
  collect_metrics()
eval_tune_xgboost

# 图示
# autoplot(tune_xgboost)
eval_tune_xgboost %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'mtry', values = ~mtry),
      list(label = 'min_n', values = ~min_n),
      list(label = 'tree_depth', values = ~tree_depth),
      list(label = 'learn_rate', values = ~learn_rate),
      list(label = 'loss_reduction', values = ~loss_reduction),
      list(label = 'sample_size', values = ~sample_size)
    )
  ) %>%
  plotly::layout(title = "xgboost HPO Guided by AUCROC")
tune_xgboost_plot <- eval_tune_xgboost %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'mtry', values = ~mtry),
      list(label = 'min_n', values = ~min_n),
      list(label = 'tree_depth', values = ~tree_depth),
      list(label = 'learn_rate', values = ~learn_rate),
      list(label = 'loss_reduction', values = ~loss_reduction),
      list(label = 'sample_size', values = ~sample_size)
    )
  ) %>%
  plotly::layout(title = "xgboost HPO Guided by AUCROC")
tune_xgboost_plot
scope <- plotly::kaleido()
scope$transform(tune_xgboost_plot, "tune_xgboost_plot.pdf")


# 经过交叉验证得到的最优超参数
hpbest_xgboost <- tune_xgboost %>%
  select_by_one_std_err(metric = "roc_auc", desc(min_n))
hpbest_xgboost





# 交叉验证结果
eval_tune_enet <- tune_enet %>%
  collect_metrics()
eval_tune_enet

# 图示
# autoplot(tune_enet)
eval_tune_enet %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'mixture', values = ~mixture),
      list(label = 'penalty', values = ~penalty)
    )
  ) %>%
  plotly::layout(title = "ENet HPO Guided by AUCROC")


tune_enet_plot <- eval_tune_enet %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'mixture', values = ~mixture),
      list(label = 'penalty', values = ~penalty)
    )
  ) %>%
  plotly::layout(title = "ENet HPO Guided by AUCROC")
tune_enet_plot
scope <- plotly::kaleido()
scope$transform(tune_enet_plot, "tune_enet_plot.pdf")



# 经过交叉验证得到的最优超参数
hpbest_enet <- tune_enet %>%
  select_by_one_std_err(metric = "roc_auc", desc(penalty))
hpbest_enet



# 交叉验证结果
eval_tune_svm <- tune_svm %>%
  collect_metrics()
eval_tune_svm

# 图示
# autoplot(tune_svm)
eval_tune_svm %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'cost', values = ~cost),
      list(label = 'rbf_sigma', values = ~rbf_sigma)
    )
  ) %>%
  plotly::layout(title = "SVM HPO Guided by AUCROC")

tune_svm_plot <- eval_tune_svm %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'cost', values = ~cost),
      list(label = 'rbf_sigma', values = ~rbf_sigma)
    )
  ) %>%
  plotly::layout(title = "SVM HPO Guided by AUCROC")
tune_svm_plot
scope <- plotly::kaleido()
scope$transform(tune_svm_plot, "tune_svm_plot.pdf")



# 经过交叉验证得到的最优超参数
hpbest_svm <- tune_svm %>%
  select_best(metric = "roc_auc")
hpbest_svm






# 交叉验证结果
eval_tune_mlp <- tune_mlp %>%
  collect_metrics()
eval_tune_mlp

# 图示
# autoplot(tune_mlp)
eval_tune_mlp %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'hidden_units', values = ~hidden_units),
      list(label = 'penalty', values = ~penalty),
      list(label = 'epochs', values = ~epochs)
    )
  ) %>%
  plotly::layout(title = "MLP HPO Guided by AUCROC")


tune_mlp_plot <- eval_tune_mlp %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'hidden_units', values = ~hidden_units),
      list(label = 'penalty', values = ~penalty),
      list(label = 'epochs', values = ~epochs)
    )
  ) %>%
  plotly::layout(title = "MLP HPO Guided by AUCROC")
tune_mlp_plot
scope <- plotly::kaleido()
scope$transform(tune_mlp_plot, "tune_mlp_plot.pdf")

# 经过交叉验证得到的最优超参数
hpbest_mlp <- tune_mlp %>%
  select_by_one_std_err(metric = "roc_auc", desc(penalty))
hpbest_mlp




# 交叉验证结果
eval_tune_lightgbm <- tune_lightgbm %>%
  collect_metrics()
eval_tune_lightgbm

# 图示
# autoplot(tune_lightgbm)
eval_tune_lightgbm %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'mtry', values = ~mtry),
      list(label = 'trees', values = ~trees),
      list(label = 'min_n', values = ~min_n),
      list(label = 'tree_depth', values = ~tree_depth),
      list(label = 'learn_rate', values = ~learn_rate),
      list(label = 'loss_reduction', values = ~loss_reduction)
    )
  ) %>%
  plotly::layout(title = "lightgbm HPO Guided by AUCROC")


tune_lightgbm_plot <- eval_tune_lightgbm %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'mtry', values = ~mtry),
      list(label = 'trees', values = ~trees),
      list(label = 'min_n', values = ~min_n),
      list(label = 'tree_depth', values = ~tree_depth),
      list(label = 'learn_rate', values = ~learn_rate),
      list(label = 'loss_reduction', values = ~loss_reduction)
    )
  ) %>%
  plotly::layout(title = "lightGBM HPO Guided by AUCROC")
tune_lightgbm_plot
scope <- plotly::kaleido()
scope$transform(tune_lightgbm_plot, "tune_lightgbm_plot.pdf")



# 经过交叉验证得到的最优超参数
hpbest_lightgbm <- tune_lightgbm %>%
  select_by_one_std_err(metric = "roc_auc", desc(min_n))
hpbest_lightgbm



# 交叉验证结果
eval_tune_knn <- tune_knn %>%
  collect_metrics()
eval_tune_knn

# 图示
# autoplot(tune_knn)
eval_tune_knn %>% 
  filter(.metric == "roc_auc") %>%
  mutate(weight_func2 = as.numeric(as.factor(weight_func))) %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'neighbors', values = ~neighbors),
      list(label = 'weight_func', values = ~weight_func2,
           range = c(1,length(unique(eval_tune_knn$weight_func))), 
           tickvals = 1:length(unique(eval_tune_knn$weight_func)),
           ticktext = sort(unique(eval_tune_knn$weight_func)))
    )
  ) %>%
  plotly::layout(title = "KNN HPO Guided by AUCROC")



tune_knn_plot <- eval_tune_knn %>% 
  filter(.metric == "roc_auc") %>%
  mutate(weight_func2 = as.numeric(as.factor(weight_func))) %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'neighbors', values = ~neighbors),
      list(label = 'weight_func', values = ~weight_func2)
    )
  ) %>%
  plotly::layout(title = "KNN HPO Guided by AUCROC")
tune_knn_plot
scope <- plotly::kaleido()
scope$transform(tune_knn_plot, "tune_knn_plot.pdf")



# 经过交叉验证得到的最优超参数
hpbest_knn <- tune_knn %>%
  select_by_one_std_err(metric = "roc_auc", desc(neighbors))
hpbest_knn






tune_knn_plot
scope <- plotly::kaleido()
scope$transform(tune_knn_plot, "tune_knn_plot.pdf")


# 经过交叉验证得到的最优超参数
hpbest_knn <- tune_knn %>%
  select_by_one_std_err(metric = "roc_auc", desc(neighbors))
hpbest_knn







# 采用最优超参数组合训练最终模型
set.seed(42)
final_dt <- wk_dt %>%
  finalize_workflow(hpbest_dt) %>%
  fit(training_set)
final_dt



set.seed(42)
final_rf <- wk_rf %>%
  finalize_workflow(hpbest_rf) %>%
  fit(training_set)
final_rf


set.seed(42)
final_xgboost <- wk_xgboost %>%
  finalize_workflow(hpbest_xgboost) %>%
  fit(training_set)
final_xgboost



# 采用最优超参数组合训练最终模型
set.seed(42)
final_enet <- wk_enet %>%
  finalize_workflow(hpbest_enet) %>%
  fit(training_set)
final_enet


# 采用最优超参数组合训练最终模型
set.seed(42)
final_svm <- wk_svm %>%
  finalize_workflow(hpbest_svm) %>%
  fit(training_set)
final_svm



# 采用最优超参数组合训练最终模型
set.seed(42)
final_mlp <- wk_mlp %>%
  finalize_workflow(hpbest_mlp) %>%
  fit(training_set)
final_mlp


# 采用最优超参数组合训练最终模型
set.seed(42)
final_lightgbm <- wk_lightgbm %>%
  finalize_workflow(hpbest_lightgbm) %>%
  fit(training_set)
final_lightgbm


# 采用最优超参数组合训练最终模型
set.seed(42)
final_knn <- wk_knn %>%
  finalize_workflow(hpbest_knn) %>%
  fit(training_set)
final_knn


set.seed(42)
final_logistic <- wk_logistic %>%
  fit(training_set)
final_logistic



# 训练集预测评估
predtrain_dt <- eval4cls2(
  model = final_dt, 
  dataset = training_set, 
  yname = "ending", 
  modelname = "DT", 
  datasetname = "Training Set",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_dt$prediction
predtrain_dt$predprobplot
predtrain_dt$rocresult
predtrain_dt$rocplot
predtrain_dt$prresult
predtrain_dt$prplot
predtrain_dt$cmresult
predtrain_dt$cmplot
predtrain_dt$metrics
predtrain_dt$diycutoff
predtrain_dt$ksplot
predtrain_dt$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_dt$proc)
pROC::ci.auc(predtrain_dt$proc)

# 预测评估测试集预测评估
predtest_dt <- eval4cls2(
  model = final_dt, 
  dataset = validation_set, 
  yname = "ending", 
  modelname = "DT", 
  datasetname = "Internal Validation Set",
  cutoff = predtrain_dt$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_dt$prediction
predtest_dt$predprobplot
predtest_dt$rocresult
predtest_dt$rocplot
predtest_dt$prresult
predtest_dt$prplot
predtest_dt$cmresult
predtest_dt$cmplot
predtest_dt$metrics
predtest_dt$diycutoff
predtest_dt$ksplot
predtest_dt$dcaplot




# 预测评估测试集2预测评估
predtest_dt2 <- eval4cls2(
  model = final_dt, 
  dataset = validation_set2, 
  yname = "ending", 
  modelname = "DT", 
  datasetname = "External Validation Set",
  cutoff = predtrain_dt$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)

predtest_dt2$prediction
predtest_dt2$predprobplot
predtest_dt2$rocresult
predtest_dt2$rocplot
predtest_dt2$prresult
predtest_dt2$prplot
predtest_dt2$cmresult
predtest_dt2$cmplot
predtest_dt2$metrics
predtest_dt2$diycutoff
predtest_dt2$ksplot
predtest_dt2$dcaplot


# pROC包auc值及其置信区间
pROC::auc(predtest_dt2$proc)
pROC::ci.auc(predtest_dt2$proc)

# ROC比较检验
pROC::roc.test(predtrain_dt$proc, predtest_dt2$proc)


# 合并训练集和测试集上ROC曲线
predtrain_dt$rocresult %>%
  bind_rows(predtest_dt$rocresult) %>%
  bind_rows(predtest_dt2$rocresult) %>%
  mutate(dataAUC = paste(data, " ROCAUC:", round(ROCAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上PR曲线
predtrain_dt$prresult %>%
  bind_rows(predtest_dt$prresult) %>%
  bind_rows(predtest_dt2$prresult) %>%
  mutate(dataAUC = paste(data, " PRAUC:", round(PRAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上性能指标
predtrain_dt$metrics %>%
  bind_rows(predtest_dt$metrics) %>%
  bind_rows(predtest_dt2$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_dt <- bestcv4cls2(
  wkflow = wk_dt,
  tuneresult = tune_dt,
  hpbest = hpbest_dt,
  yname = "ending",
  modelname = "DT",
  v = 5,
  positivelevel = yourpositivelevel
)
evalcv_dt$cvroc
evalcv_dt$cvpr
evalcv_dt$evalcv

# 保存评估结果
save(datarecipe_dt,
     model_dt,
     wk_dt,
     hpgrid_dt, # 如果采用贝叶斯优化则无需保存
     tune_dt,
     predtrain_dt,
     predtest_dt,
     predtest_dt2,
     evalcv_dt,
     file = ".\\cls2\\evalresult_dt.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_dt_heart <- final_dt
traindata_heart <- training_set
save(final_dt_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_dt_heart.RData")




# 训练集预测评估
predtrain_rf <- eval4cls2(
  model = final_rf, 
  dataset = training_set, 
  yname = "ending", 
  modelname = "RF", 
  datasetname = "Training Set",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_rf$prediction
predtrain_rf$predprobplot
predtrain_rf$rocresult
predtrain_rf$rocplot
predtrain_rf$prresult
predtrain_rf$prplot
predtrain_rf$cmresult
predtrain_rf$cmplot
predtrain_rf$metrics
predtrain_rf$diycutoff
predtrain_rf$ksplot
predtrain_rf$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_rf$proc)
pROC::ci.auc(predtrain_rf$proc)

# 预测评估测试集预测评估
predtest_rf <- eval4cls2(
  model = final_rf, 
  dataset = validation_set, 
  yname = "ending", 
  modelname = "RF", 
  datasetname = "Internal Validation Set",
  cutoff = predtrain_rf$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)


predtest_rf$prediction
predtest_rf$predprobplot
predtest_rf$rocresult
predtest_rf$rocplot
predtest_rf$prresult
predtest_rf$prplot
predtest_rf$cmresult
predtest_rf$cmplot
predtest_rf$metrics
predtest_rf$diycutoff
predtest_rf$ksplot
predtest_rf$dcaplot

# 预测评估外部验证集集预测评估
predtest_rf2 <- eval4cls2(
  model = final_rf, 
  dataset = validation_set2, 
  yname = "ending", 
  modelname = "RF", 
  datasetname = "External Validation set",
  cutoff = predtrain_rf$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)

# pROC包auc值及其置信区间
pROC::auc(predtest_rf$proc)
pROC::ci.auc(predtest_rf$proc)

# ROC比较检验
pROC::roc.test(predtest_rf$proc, predtest_rf2$proc)


# 合并训练集和测试集上ROC曲线
predtrain_rf$rocresult %>%
  bind_rows(predtest_rf$rocresult) %>%
  bind_rows(predtest_rf2$rocresult) %>%
  mutate(dataAUC = paste(data, " ROCAUC:", round(ROCAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上PR曲线
predtrain_rf$prresult %>%
  bind_rows(predtest_rf$prresult) %>%
  bind_rows(predtest_rf2$prresult) %>%
  mutate(dataAUC = paste(data, " PRAUC:", round(PRAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上性能指标
predtrain_rf$metrics %>%
  bind_rows(predtest_rf$metrics) %>%
  bind_rows(predtest_rf2$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_rf <- bestcv4cls2(
  wkflow = wk_rf,
  tuneresult = tune_rf,
  hpbest = hpbest_rf,
  yname = "ending",
  modelname = "RF",
  v = 5,
  positivelevel = yourpositivelevel
)
evalcv_rf$cvroc
evalcv_rf$cvpr
evalcv_rf$evalcv

# 保存评估结果
save(datarecipe_rf,
     model_rf,
     wk_rf,
     hpgrid_rf,   # 如果采用贝叶斯优化则替换为 param_rf
     tune_rf,
     predtrain_rf,
     predtest_rf,
     predtest_rf2,
     evalcv_rf,
     file = ".\\cls2\\evalresult_rf.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_rf_heart <- final_rf
traindata_heart <- training_set
save(final_rf_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_rf_heart.RData")










# 训练集预测评估
predtrain_xgboost <- eval4cls2(
  model = final_xgboost, 
  dataset = training_set, 
  yname = "ending", 
  modelname = "Xgboost", 
  datasetname = "Training Set",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_xgboost$prediction
predtrain_xgboost$predprobplot
predtrain_xgboost$rocresult
predtrain_xgboost$rocplot
predtrain_xgboost$prresult
predtrain_xgboost$prplot
predtrain_xgboost$cmresult
predtrain_xgboost$cmplot
predtrain_xgboost$metrics
predtrain_xgboost$diycutoff
predtrain_xgboost$ksplot
predtrain_xgboost$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_xgboost$proc)
pROC::ci.auc(predtrain_xgboost$proc)

# 预测评估测试集预测评估
predtest_xgboost <- eval4cls2(
  model = final_xgboost, 
  dataset = validation_set, 
  yname = "ending", 
  modelname = "Xgboost", 
  datasetname = "Internal Validation Set",
  cutoff = predtrain_xgboost$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_xgboost$prediction
predtest_xgboost$predprobplot
predtest_xgboost$rocresult
predtest_xgboost$rocplot
predtest_xgboost$prresult
predtest_xgboost$prplot
predtest_xgboost$cmresult
predtest_xgboost$cmplot
predtest_xgboost$metrics
predtest_xgboost$diycutoff
predtest_xgboost$ksplot
predtest_xgboost$dcaplot

# 预测评估测试集预测评估
predtest_xgboost2 <- eval4cls2(
  model = final_xgboost, 
  dataset = validation_set2, 
  yname = "ending", 
  modelname = "Xgboost", 
  datasetname = "External Validation set",
  cutoff = predtrain_xgboost$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_xgboost2$prediction
predtest_xgboost2$predprobplot
predtest_xgboost2$rocresult
predtest_xgboost2$rocplot
predtest_xgboost2$prresult
predtest_xgboost2$prplot
predtest_xgboost2$cmresult
predtest_xgboost2$cmplot
predtest_xgboost2$metrics
predtest_xgboost2$diycutoff
predtest_xgboost2$ksplot
predtest_xgboost2$dcaplot


# pROC包auc值及其置信区间
pROC::auc(predtest_xgboost$proc)
pROC::ci.auc(predtest_xgboost$proc)

# ROC比较检验
pROC::roc.test(predtest_xgboost$proc, predtest_xgboost2$proc)


# 合并训练集和测试集上ROC曲线
predtrain_xgboost$rocresult %>%
  bind_rows(predtest_xgboost$rocresult) %>%
  bind_rows(predtest_xgboost2$rocresult) %>%
  mutate(dataAUC = paste(data, " ROCAUC:", round(ROCAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上PR曲线
predtrain_xgboost$prresult %>%
  bind_rows(predtest_xgboost$prresult) %>%
  bind_rows(predtest_xgboost2$prresult) %>%
  mutate(dataAUC = paste(data, " PRAUC:", round(PRAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上性能指标
predtrain_xgboost$metrics %>%
  bind_rows(predtest_xgboost$metrics) %>%
  bind_rows(predtest_xgboost2$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_xgboost <- bestcv4cls2(
  wkflow = wk_xgboost,
  tuneresult = tune_xgboost,
  hpbest = hpbest_xgboost,
  yname = "ending",
  modelname = "Xgboost",
  v = 5,
  positivelevel = yourpositivelevel
)
evalcv_xgboost$cvroc
evalcv_xgboost$cvpr
evalcv_xgboost$evalcv

# 保存评估结果
save(datarecipe_xgboost,
     model_xgboost,
     wk_xgboost,
     hpgrid_xgboost,  # 如果采用贝叶斯优化则替换为 param_xgboost
     tune_xgboost,
     predtrain_xgboost,
     predtest_xgboost,
     predtest_xgboost2,
     evalcv_xgboost,
     file = ".\\cls2\\evalresult_xgboost.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_xgboost_heart <- final_xgboost
traindata_heart <- training_set
save(final_xgboost_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_xgboost_heart.RData")







# 训练集预测评估
predtrain_enet <- eval4cls2(
  model = final_enet, 
  dataset = training_set, 
  yname = "ending", 
  modelname = "ENet", 
  datasetname = "Training Set",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_enet$prediction
predtrain_enet$predprobplot
predtrain_enet$rocresult
predtrain_enet$rocplot
predtrain_enet$prresult
predtrain_enet$prplot
predtrain_enet$cmresult
predtrain_enet$cmplot
predtrain_enet$metrics
predtrain_enet$diycutoff
predtrain_enet$ksplot
predtrain_enet$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_enet$proc)
pROC::ci.auc(predtrain_enet$proc)

# 预测评估测试集预测评估
predtest_enet <- eval4cls2(
  model = final_enet, 
  dataset = validation_set, 
  yname = "ending", 
  modelname = "ENet", 
  datasetname = "Internal Validation Set",
  cutoff = predtrain_enet$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_enet$prediction
predtest_enet$predprobplot
predtest_enet$rocresult
predtest_enet$rocplot
predtest_enet$prresult
predtest_enet$prplot
predtest_enet$cmresult
predtest_enet$cmplot
predtest_enet$metrics
predtest_enet$diycutoff
predtest_enet$ksplot
predtest_enet$dcaplot

# 预测评估外部验证集预测评估
predtest_enet2 <- eval4cls2(
  model = final_enet, 
  dataset = validation_set2, 
  yname = "ending", 
  modelname = "ENet", 
  datasetname = "External Validation Set",
  cutoff = predtrain_enet$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)

# pROC包auc值及其置信区间
pROC::auc(predtest_enet$proc)
pROC::ci.auc(predtest_enet$proc)

# ROC比较检验
pROC::roc.test(predtest_enet$proc, predtest_enet2$proc)


# 合并训练集和测试集上ROC曲线
predtrain_enet$rocresult %>%
  bind_rows(predtest_enet$rocresult) %>%
  bind_rows(predtest_enet2$rocresult) %>%
  mutate(dataAUC = paste(data, " ROCAUC:", round(ROCAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上PR曲线
predtrain_enet$prresult %>%
  bind_rows(predtest_enet$prresult) %>%
  bind_rows(predtest_enet2$prresult) %>%
  mutate(dataAUC = paste(data, " PRAUC:", round(PRAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上性能指标
predtrain_enet$metrics %>%
  bind_rows(predtest_enet$metrics) %>%
  bind_rows(predtest_enet2$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_enet <- bestcv4cls2(
  wkflow = wk_enet,
  tuneresult = tune_enet,
  hpbest = hpbest_enet,
  yname = "ending",
  modelname = "ENet",
  v = 5,
  positivelevel = yourpositivelevel
)
evalcv_enet$cvroc
evalcv_enet$cvpr
evalcv_enet$evalcv

# 保存评估结果
save(datarecipe_enet,
     model_enet,
     wk_enet,
     hpgrid_enet,   # 如果采用贝叶斯优化则无需保存
     tune_enet,
     predtrain_enet,
     predtest_enet,
     predtest_enet2,
     evalcv_enet,
     file = ".\\cls2\\evalresult_enet.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_enet_heart <- final_enet
traindata_heart <- training_set
save(final_enet_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_enet_heart.RData")


# 训练集预测评估
predtrain_svm <- eval4cls2(
  model = final_svm, 
  dataset = training_set, 
  yname = "ending", 
  modelname = "SVM", 
  datasetname = "Training Set",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_svm$prediction
predtrain_svm$predprobplot
predtrain_svm$rocresult
predtrain_svm$rocplot
predtrain_svm$prresult
predtrain_svm$prplot
predtrain_svm$cmresult
predtrain_svm$cmplot
predtrain_svm$metrics
predtrain_svm$diycutoff
predtrain_svm$ksplot
predtrain_svm$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_svm$proc)
pROC::ci.auc(predtrain_svm$proc)

# 预测评估测试集预测评估
predtest_svm <- eval4cls2(
  model = final_svm, 
  dataset = validation_set, 
  yname = "ending", 
  modelname = "SVM", 
  datasetname = "Internal Validation Set",
  cutoff = predtrain_svm$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_svm$prediction
predtest_svm$predprobplot
predtest_svm$rocresult
predtest_svm$rocplot
predtest_svm$prresult
predtest_svm$prplot
predtest_svm$cmresult
predtest_svm$cmplot
predtest_svm$metrics
predtest_svm$diycutoff
predtest_svm$ksplot
predtest_svm$dcaplot

# 预测评估测试集预测评估
predtest_svm2 <- eval4cls2(
  model = final_svm, 
  dataset = validation_set2, 
  yname = "ending", 
  modelname = "SVM", 
  datasetname = "External Validation Set",
  cutoff = predtrain_svm$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)

# pROC包auc值及其置信区间
pROC::auc(predtest_svm$proc)
pROC::ci.auc(predtest_svm$proc)

# ROC比较检验
pROC::roc.test(predtest_svm$proc, predtest_svm2$proc)

# 合并训练集和测试集上ROC曲线
predtrain_svm$rocresult %>%
  bind_rows(predtest_svm$rocresult) %>%
  bind_rows(predtest_svm2$rocresult) %>%
  mutate(dataAUC = paste(data, " ROCAUC:", round(ROCAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上PR曲线
predtrain_svm$prresult %>%
  bind_rows(predtest_svm$prresult) %>%
  bind_rows(predtest_svm2$prresult) %>%
  mutate(dataAUC = paste(data, " PRAUC:", round(PRAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上性能指标
predtrain_svm$metrics %>%
  bind_rows(predtest_svm$metrics) %>%
  bind_rows(predtest_svm2$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_svm <- bestcv4cls2(
  wkflow = wk_svm,
  tuneresult = tune_svm,
  hpbest = hpbest_svm,
  yname = "ending",
  modelname = "SVM",
  v = 5,
  positivelevel = yourpositivelevel
)
evalcv_svm$cvroc
evalcv_svm$cvpr
evalcv_svm$evalcv

# 保存评估结果
save(datarecipe_svm,
     model_svm,
     wk_svm,
     hpgrid_svm, # 如果采用贝叶斯优化则无需保存
     tune_svm,
     predtrain_svm,
     predtest_svm,
     predtest_svm2,
     evalcv_svm,
     file = ".\\cls2\\evalresult_svm.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_svm_heart <- final_svm
traindata_heart <- training_set
save(final_svm_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_svm_heart.RData")







# 训练集预测评估
predtrain_mlp <- eval4cls2(
  model = final_mlp, 
  dataset = training_set, 
  yname = "ending", 
  modelname = "MLP", 
  datasetname = "Training Set",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_mlp$prediction
predtrain_mlp$predprobplot
predtrain_mlp$rocresult
predtrain_mlp$rocplot
predtrain_mlp$prresult
predtrain_mlp$prplot
predtrain_mlp$cmresult
predtrain_mlp$cmplot
predtrain_mlp$metrics
predtrain_mlp$diycutoff
predtrain_mlp$ksplot
predtrain_mlp$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_mlp$proc)
pROC::ci.auc(predtrain_mlp$proc)

# 预测评估测试集预测评估
predtest_mlp <- eval4cls2(
  model = final_mlp, 
  dataset = validation_set, 
  yname = "ending", 
  modelname = "MLP", 
  datasetname = "Internal Validation Set",
  cutoff = predtrain_mlp$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_mlp$prediction
predtest_mlp$predprobplot
predtest_mlp$rocresult
predtest_mlp$rocplot
predtest_mlp$prresult
predtest_mlp$prplot
predtest_mlp$cmresult
predtest_mlp$cmplot
predtest_mlp$metrics
predtest_mlp$diycutoff
predtest_mlp$ksplot
predtest_mlp$dcaplot

# 预测评估外部验证集集预测评估
predtest_mlp2 <- eval4cls2(
  model = final_mlp, 
  dataset = validation_set2, 
  yname = "ending", 
  modelname = "MLP", 
  datasetname = "External Validation Set",
  cutoff = predtrain_mlp$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_mlp$prediction
predtest_mlp$predprobplot
predtest_mlp$rocresult
predtest_mlp$rocplot
predtest_mlp$prresult
predtest_mlp$prplot
predtest_mlp$cmresult
predtest_mlp$cmplot
predtest_mlp$metrics
predtest_mlp$diycutoff
predtest_mlp$ksplot
predtest_mlp$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtest_mlp$proc)
pROC::ci.auc(predtest_mlp$proc)

# ROC比较检验
pROC::roc.test(predtest_mlp$proc, predtest_mlp2$proc)

# 合并训练集和测试集上ROC曲线
predtrain_mlp$rocresult %>%
  bind_rows(predtest_mlp$rocresult) %>%
  bind_rows(predtest_mlp2$rocresult) %>%
  mutate(dataAUC = paste(data, " ROCAUC:", round(ROCAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上PR曲线
predtrain_mlp$prresult %>%
  bind_rows(predtest_mlp$prresult) %>%
  bind_rows(predtest_mlp2$prresult) %>%
  mutate(dataAUC = paste(data, " PRAUC:", round(PRAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上性能指标
predtrain_mlp$metrics %>%
  bind_rows(predtest_mlp$metrics) %>%
  bind_rows(predtest_mlp2$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_mlp <- bestcv4cls2(
  wkflow = wk_mlp,
  tuneresult = tune_mlp,
  hpbest = hpbest_mlp,
  yname = "ending",
  modelname = "MLP",
  v = 5,
  positivelevel = yourpositivelevel
)
evalcv_mlp$cvroc
evalcv_mlp$cvpr
evalcv_mlp$evalcv

# 保存评估结果
save(datarecipe_mlp,
     model_mlp,
     wk_mlp,
     hpgrid_mlp, # 如果采用贝叶斯优化则无需保存
     tune_mlp,
     predtrain_mlp,
     predtest_mlp,
     predtest_mlp2,
     evalcv_mlp,
     file = ".\\cls2\\evalresult_mlp.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_mlp_heart <- final_mlp
traindata_heart <- training_set
save(final_mlp_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_mlp_heart.RData")







# 训练集预测评估
predtrain_lightgbm <- eval4cls2(
  model = final_lightgbm, 
  dataset = training_set, 
  yname = "ending", 
  modelname = "Lightgbm", 
  datasetname = "Training Set",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_lightgbm$prediction
predtrain_lightgbm$predprobplot
predtrain_lightgbm$rocresult
predtrain_lightgbm$rocplot
predtrain_lightgbm$prresult
predtrain_lightgbm$prplot
predtrain_lightgbm$cmresult
predtrain_lightgbm$cmplot
predtrain_lightgbm$metrics
predtrain_lightgbm$diycutoff
predtrain_lightgbm$ksplot
predtrain_lightgbm$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_lightgbm$proc)
pROC::ci.auc(predtrain_lightgbm$proc)

# 预测评估测试集预测评估
predtest_lightgbm <- eval4cls2(
  model = final_lightgbm, 
  dataset = validation_set, 
  yname = "ending", 
  modelname = "Lightgbm", 
  datasetname = "Internal Validation Set",
  cutoff = predtrain_lightgbm$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_lightgbm$prediction
predtest_lightgbm$predprobplot
predtest_lightgbm$rocresult
predtest_lightgbm$rocplot
predtest_lightgbm$prresult
predtest_lightgbm$prplot
predtest_lightgbm$cmresult
predtest_lightgbm$cmplot
predtest_lightgbm$metrics
predtest_lightgbm$diycutoff
predtest_lightgbm$ksplot
predtest_lightgbm$dcaplot

# 预测评估测试集预测评估
predtest_lightgbm2 <- eval4cls2(
  model = final_lightgbm, 
  dataset = validation_set2, 
  yname = "ending", 
  modelname = "Lightgbm", 
  datasetname = "External Validation Set",
  cutoff = predtrain_lightgbm$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_lightgbm$prediction
predtest_lightgbm$predprobplot
predtest_lightgbm$rocresult
predtest_lightgbm$rocplot
predtest_lightgbm$prresult
predtest_lightgbm$prplot
predtest_lightgbm$cmresult
predtest_lightgbm$cmplot
predtest_lightgbm$metrics
predtest_lightgbm$diycutoff
predtest_lightgbm$ksplot
predtest_lightgbm$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtest_lightgbm$proc)
pROC::ci.auc(predtest_lightgbm$proc)

# ROC比较检验
pROC::roc.test(predtest_lightgbm$proc, predtest_lightgbm2$proc)


# 合并训练集和测试集上ROC曲线
predtrain_lightgbm$rocresult %>%
  bind_rows(predtest_lightgbm$rocresult) %>%
  bind_rows(predtest_lightgbm2$rocresult) %>%
  mutate(dataAUC = paste(data, " ROCAUC:", round(ROCAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上PR曲线
predtrain_lightgbm$prresult %>%
  bind_rows(predtest_lightgbm$prresult) %>%
  bind_rows(predtest_lightgbm2$prresult) %>%
  mutate(dataAUC = paste(data, " PRAUC:", round(PRAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上性能指标
predtrain_lightgbm$metrics %>%
  bind_rows(predtest_lightgbm$metrics) %>%
  bind_rows(predtest_lightgbm2$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_lightgbm <- bestcv4cls2(
  wkflow = wk_lightgbm,
  tuneresult = tune_lightgbm,
  hpbest = hpbest_lightgbm,
  yname = "ending",
  modelname = "Lightgbm",
  v = 5,
  positivelevel = yourpositivelevel
)
evalcv_lightgbm$cvroc
evalcv_lightgbm$cvpr
evalcv_lightgbm$evalcv

# 保存评估结果
save(datarecipe_lightgbm,
     model_lightgbm,
     wk_lightgbm,
     hpgrid_lightgbm, # 如果采用贝叶斯优化则无需保存
     tune_lightgbm,
     predtrain_lightgbm,
     predtest_lightgbm,
     predtest_lightgbm2,
     evalcv_lightgbm,
     file = ".\\cls2\\evalresult_lightgbm.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_lightgbm_heart <- final_lightgbm
traindata_heart <- training_set
save(final_lightgbm_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_lightgbm_heart.RData")







# 训练集预测评估
predtrain_knn <- eval4cls2(
  model = final_knn, 
  dataset = training_set, 
  yname = "ending", 
  modelname = "KNN", 
  datasetname = "Training Set",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_knn$prediction
predtrain_knn$predprobplot
predtrain_knn$rocresult
predtrain_knn$rocplot
predtrain_knn$prresult
predtrain_knn$prplot
predtrain_knn$cmresult
predtrain_knn$cmplot
predtrain_knn$metrics
predtrain_knn$diycutoff
predtrain_knn$ksplot
predtrain_knn$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_knn$proc)
pROC::ci.auc(predtrain_knn$proc)

# 预测评估测试集预测评估
predtest_knn <- eval4cls2(
  model = final_knn, 
  dataset = validation_set, 
  yname = "ending", 
  modelname = "KNN", 
  datasetname = "Internal Validation Set",
  cutoff = predtrain_knn$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_knn$prediction
predtest_knn$predprobplot
predtest_knn$rocresult
predtest_knn$rocplot
predtest_knn$prresult
predtest_knn$prplot
predtest_knn$cmresult
predtest_knn$cmplot
predtest_knn$metrics
predtest_knn$diycutoff
predtest_knn$ksplot
predtest_knn$dcaplot

# 预测评估测试集预测评估
predtest_knn2 <- eval4cls2(
  model = final_knn, 
  dataset = validation_set2, 
  yname = "ending", 
  modelname = "KNN", 
  datasetname = "External Validation Set",
  cutoff = predtrain_knn$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)

# pROC包auc值及其置信区间
pROC::auc(predtest_knn$proc)
pROC::ci.auc(predtest_knn$proc)

# ROC比较检验
pROC::roc.test(predtest_knn$proc, predtest_knn2$proc)

# 合并训练集和测试集上ROC曲线
predtrain_knn$rocresult %>%
  bind_rows(predtest_knn$rocresult) %>%
  bind_rows(predtest_knn2$rocresult) %>%
  mutate(dataAUC = paste(data, " ROCAUC:", round(ROCAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上PR曲线
predtrain_knn$prresult %>%
  bind_rows(predtest_knn$prresult) %>%
  bind_rows(predtest_knn2$prresult) %>%
  mutate(dataAUC = paste(data, " PRAUC:", round(PRAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上性能指标
predtrain_knn$metrics %>%
  bind_rows(predtest_knn$metrics) %>%
  bind_rows(predtest_knn2$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_knn <- bestcv4cls2(
  wkflow = wk_knn,
  tuneresult = tune_knn,
  hpbest = hpbest_knn,
  yname = "ending",
  modelname = "KNN",
  v = 5,
  positivelevel = yourpositivelevel
)
evalcv_knn$cvroc
evalcv_knn$cvpr
evalcv_knn$evalcv

# 保存评估结果
save(datarecipe_knn,
     model_knn,
     wk_knn,
     hpgrid_knn, # 如果采用贝叶斯优化则无需保存
     tune_knn,
     predtrain_knn,
     predtest_knn,
     predtest_knn2,
     evalcv_knn,
     file = ".\\cls2\\evalresult_knn.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_knn_heart <- final_knn
traindata_heart <- training_set
save(final_knn_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_knn_heart.RData")





# 训练集预测评估
predtrain_logistic <- eval4cls2(
  model = final_logistic, 
  dataset = training_set, 
  yname = "ending", 
  modelname = "Logistic", 
  datasetname = "Training Set",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_logistic$prediction
predtrain_logistic$predprobplot
predtrain_logistic$rocresult
predtrain_logistic$rocplot
predtrain_logistic$prresult
predtrain_logistic$prplot
predtrain_logistic$cmresult
predtrain_logistic$cmplot
predtrain_logistic$metrics
predtrain_logistic$diycutoff
predtrain_logistic$ksplot
predtrain_logistic$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_logistic$proc)
pROC::ci.auc(predtrain_logistic$proc)

# 预测评估测试集预测评估
predtest_logistic <- eval4cls2(
  model = final_logistic, 
  dataset = validation_set, 
  yname = "ending", 
  modelname = "Logistic", 
  datasetname = "Internal Validation Set",
  cutoff = predtrain_logistic$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_logistic$prediction
predtest_logistic$predprobplot
predtest_logistic$rocresult
predtest_logistic$rocplot
predtest_logistic$prresult
predtest_logistic$prplot
predtest_logistic$cmresult
predtest_logistic$cmplot
predtest_logistic$metrics
predtest_logistic$diycutoff
predtest_logistic$ksplot
predtest_logistic$dcaplot

# 预测评估测试集预测评估
predtest_logistic2 <- eval4cls2(
  model = final_logistic, 
  dataset = validation_set2, 
  yname = "ending", 
  modelname = "Logistic", 
  datasetname = "External Validation Set",
  cutoff = predtrain_logistic$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)

# pROC包auc值及其置信区间
pROC::auc(predtest_logistic$proc)
pROC::ci.auc(predtest_logistic$proc)

# ROC比较检验
pROC::roc.test(predtest_logistic$proc, predtest_logistic2$proc)

# 合并训练集和测试集上ROC曲线
predtrain_logistic$rocresult %>%
  bind_rows(predtest_logistic$rocresult) %>%
  bind_rows(predtest_logistic2$rocresult) %>%
  mutate(dataAUC = paste(data, " ROCAUC:", round(ROCAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上PR曲线
predtrain_logistic$prresult %>%
  bind_rows(predtest_logistic$prresult) %>%
  bind_rows(predtest_logistic2$prresult) %>%
  mutate(dataAUC = paste(data, " PRAUC:", round(PRAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上性能指标
predtrain_logistic$metrics %>%
  bind_rows(predtest_logistic$metrics) %>%
  bind_rows(predtest_logistic2$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

##################################################################

# 交叉验证
set.seed(42)
cv_logistic <- 
  wk_logistic %>%
  fit_resamples(
    folds,
    metrics = metric_set(yardstick::accuracy, 
                         yardstick::roc_auc, 
                         yardstick::pr_auc),
    control = control_resamples(save_pred = T,
                                verbose = T,
                                event_level = "second",
                                parallel_over = "everything",
                                save_workflow = T)
  )
cv_logistic

# 交叉验证指标结果
evalcv_logistic <- list()
# 评估指标设定
metrictemp <- metric_set(yardstick::roc_auc, yardstick::pr_auc)
evalcv_logistic$evalcv <- 
  collect_predictions(cv_logistic) %>%
  group_by(id) %>%
  metrictemp(ending, .pred_Yes, event_level = "second") %>%
  group_by(.metric) %>%
  mutate(model = "logistic",
         mean = mean(.estimate),
         sd = sd(.estimate)/sqrt(length(folds$splits)))
evalcv_logistic$evalcv

# 交叉验证预测结果图示
# ROC
evalcv_logistic$cvroc <- 
  collect_predictions(cv_logistic) %>%
  group_by(id) %>%
  roc_curve(ending, .pred_Yes, event_level = "second") %>%
  ungroup() %>%
  left_join(evalcv_logistic$evalcv %>% filter(.metric == "roc_auc"), 
            by = "id") %>%
  mutate(idAUC = paste(id, " ROCAUC:", round(.estimate, 4)),
         idAUC = forcats::as_factor(idAUC)) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = idAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())
evalcv_logistic$cvroc

# PR
evalcv_logistic$cvpr <- 
  collect_predictions(cv_logistic) %>%
  group_by(id) %>%
  pr_curve(ending, .pred_Yes, event_level = "second") %>%
  ungroup() %>%
  left_join(evalcv_logistic$evalcv %>% filter(.metric == "pr_auc"), 
            by = "id") %>%
  mutate(idAUC = paste(id, " PRAUC:", round(.estimate, 4)),
         idAUC = forcats::as_factor(idAUC)) %>%
  ggplot(aes(x = recall, y = precision, color = idAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", intercept = 1, slope = -1) +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank())
evalcv_logistic$cvpr

##################################################################

# 保存评估结果
save(datarecipe_logistic,
     model_logistic,
     wk_logistic,
     cv_logistic,
     predtrain_logistic,
     predtest_logistic,
     predtest_logistic2,
     evalcv_logistic,
     file = ".\\cls2\\evalresult_logistic.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_logistic_heart <- final_logistic
traindata_heart <- training_set
save(final_logistic_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_logistic_heart.RData")

#####################################################
# 模型机器---R语言tidymodels包机器学习分类与回归模型---二分类---模型比较

#############################################################
# remotes::install_github("tidymodels/probably")
library(tidymodels)

# 加载各个模型的评估结果
evalfiles <- list.files(".\\cls2\\", full.names = T)
lapply(evalfiles, load, .GlobalEnv)

# 横向比较的模型个数
nmodels <- 10
cols4model <- rainbow(nmodels)  # 模型统一配色
#############################################################

# 各个模型在测试集上的性能指标
predtrain_dt$metrics
eval_train <- bind_rows(
  lapply(list(predtrain_logistic, predtrain_dt, predtrain_enet,
              predtrain_knn, predtrain_rf,predtrain_svm, 
              predtrain_xgboost,predtrain_mlp,
              predtrain_lightgbm), 
         "[[", 
         "metrics")
) %>%
  mutate(model = forcats::as_factor(model))
eval_train



# 平行线图
eval_train_max <-   eval_train %>% 
  group_by(.metric) %>%
  slice_max(.estimate)
eval_train_min <-   eval_train %>% 
  group_by(.metric) %>%
  slice_min(.estimate)

eval_train %>%
  ggplot(aes(x = .metric, y = .estimate, color = model)) +
  geom_point() +
  geom_line(aes(group = model)) +
  # ggrepel::geom_text_repel(eval_max, 
  #                          mapping = aes(label = model), 
  #                          nudge_y = 0.05,
  #                          angle = 90,
  #                          show.legend = F) +
  # ggrepel::geom_text_repel(eval_min, 
  #                          mapping = aes(label = model), 
  #                          nudge_y = -0.05,
  #                          angle = 90,
  #                          show.legend = F) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/训练集平行线图.png")

# 指标热图
eval_train %>%
  dplyr::select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(model = reorder(model, roc_auc)) %>%
  pivot_longer(cols = -1) %>%
  group_by(name) %>%
  mutate(valuescale = (value-min(value)) / (max(value)-min(value))) %>%
  ungroup() %>%
  ggplot(aes(x = name, y = model, fill = valuescale)) +
  geom_tile(color = "white", show.legend = F) +
  geom_text(aes(label = round(value, 2))) +
  scale_fill_gradient(low = "green", high = "red") +
  labs(x = "", y = "", fill = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, 
                                   vjust = 1,
                                   hjust = 1))
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/训练集热图.png")


# 各个模型在测试集上的性能指标表格
eval_train2 <- eval_train %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)
eval_train2

# 各个模型在测试集上的性能指标图示
# ROCAUC
eval_train2 %>%
  ggplot(aes(x = model, y = roc_auc, fill = model)) +
  geom_col(width = 0.5, show.legend = F) +
  geom_text(aes(label = round(roc_auc, 2)), 
            nudge_y = -0.03) +
  scale_fill_manual(values = cols4model) +
  theme_bw()
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/训练集ROC柱状图.png")
#############################################################

# 各个模型在测试集上的预测概率
predtrain <- bind_rows(
  lapply(list(predtrain_logistic,predtrain_dt, predtrain_enet,
              predtrain_knn,  predtrain_rf,predtrain_svm, 
              predtrain_xgboost, predtrain_mlp,
              predtrain_lightgbm), 
         "[[", 
         "prediction")
) %>%
  mutate(model = forcats::as_factor(model))
predtrain

# 各个模型在测试集上的ROC
predtrain %>%
  group_by(model) %>%
  roc_curve(.obs, .pred_Yes, event_level = "second") %>%
  left_join(eval_train2[, c("model", "roc_auc")]) %>%
  mutate(modelauc = paste0(model, 
                           ", ROCAUC=", round(roc_auc, 4)),
         modelauc = forcats::as_factor(modelauc)) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = modelauc)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(color = "", title = paste0("ROCs on Training Set")) +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/训练集ROC曲线图.png")

# 各个模型在测试集上的PR
predtrain %>%
  group_by(model) %>%
  pr_curve(.obs, .pred_Yes, event_level = "second") %>%
  left_join(eval_train2[, c("model", "pr_auc")]) %>%
  mutate(modelauc = paste0(model, 
                           ", PRAUC=", round(pr_auc, 4)),
         modelauc = forcats::as_factor(modelauc)) %>%
  ggplot(aes(x = recall, y = precision, color = modelauc)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(color = "", title = paste0("PRs on Training Set")) +
  theme_bw() +
  theme(legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank())
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/训练集PR曲线图.png")
#############################################################

# 各个模型在测试集上的预测概率---宽数据
predtrain2 <- predtrain %>%
  dplyr::select(-.pred_No) %>%
  mutate(id = rep(1:nrow(predtrain_logistic$prediction), 
                  length(unique(predtrain$model)))) %>%
  pivot_wider(id_cols = c(id, .obs), 
              names_from = model, 
              values_from = .pred_Yes) %>%
  dplyr::select(id, .obs, sort(unique(predtrain$model)))
predtrain2

#############################################################


# 各个模型在测试集上的校准曲线
# 校准曲线附加置信区间
library(probably)
# 算法1
predtrain %>%
  cal_plot_breaks(.obs, 
                  .pred_Yes, 
                  event_level = "second", 
                  num_breaks = 5,  # 可以改大改小
                  .by = model) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(legend.position = "none")
# 算法2
predtrain %>%
  cal_plot_windowed(.obs, 
                    .pred_Yes, 
                    event_level = "second", 
                    window_size = 0.5,  # 可以改大改小
                    .by = model) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(legend.position = "none")
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/训练集校准曲线图.png")
# brier_score
bs <- predtrain %>%
  group_by(model) %>%
  yardstick::brier_class(.obs, .pred_No) %>%
  mutate(meanpred = 0.8,
         meanobs = 0.25,
         text = paste0("BS: ", round(.estimate, 3)))
# 附加bs
predtrain %>%
  cal_plot_windowed(.obs, 
                    .pred_Yes, 
                    event_level = "second", 
                    window_size = 0.5,
                    .by = model) +
  geom_text(
    bs,
    mapping = aes(x = meanpred, y = meanobs, label = text)
  ) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(legend.position = "none")
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/训练集BS曲线图.png")


#############################################################

# 各个模型在测试集上的DCA
dca_obj <- dcurves::dca(as.formula(
  paste0(".obs ~ ", 
         paste(colnames(predtrain2)[3:ncol(predtrain2)], 
               collapse = " + "))
),
data = predtrain2,
thresholds = seq(0, 1, by = 0.01)
)
plot(dca_obj, smooth = F, span = 0.5) +
  scale_color_manual(values = c("black", "grey", cols4model)) +
  labs(title = "DCA on Training Set")
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/训练集DCA曲线图.png")
#############################################################

# 各个模型在内部验证集上的性能指标
predtest_dt$metrics
eval_test <- bind_rows(
  lapply(list(predtest_logistic, predtest_dt, predtest_enet,
              predtest_knn, predtest_rf,predtest_svm, 
              predtest_xgboost,predtest_mlp,
              predtest_lightgbm), 
         "[[", 
         "metrics")
) %>%
  mutate(model = forcats::as_factor(model))
eval_test



# 平行线图
eval_test_max <-   eval_test %>% 
  group_by(.metric) %>%
  slice_max(.estimate)
eval_test_min <-   eval_test %>% 
  group_by(.metric) %>%
  slice_min(.estimate)

eval_test %>%
  ggplot(aes(x = .metric, y = .estimate, color = model)) +
  geom_point() +
  geom_line(aes(group = model)) +
  # ggrepel::geom_text_repel(eval_max, 
  #                          mapping = aes(label = model), 
  #                          nudge_y = 0.05,
  #                          angle = 90,
  #                          show.legend = F) +
  # ggrepel::geom_text_repel(eval_min, 
  #                          mapping = aes(label = model), 
  #                          nudge_y = -0.05,
  #                          angle = 90,
  #                          show.legend = F) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/内部验证集平行线图.png")

# 指标热图
eval_test %>%
  dplyr::select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(model = reorder(model, roc_auc)) %>%
  pivot_longer(cols = -1) %>%
  group_by(name) %>%
  mutate(valuescale = (value-min(value)) / (max(value)-min(value))) %>%
  ungroup() %>%
  ggplot(aes(x = name, y = model, fill = valuescale)) +
  geom_tile(color = "white", show.legend = F) +
  geom_text(aes(label = round(value, 2))) +
  scale_fill_gradient(low = "green", high = "red") +
  labs(x = "", y = "", fill = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, 
                                   vjust = 1,
                                   hjust = 1))
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/内部验证集热图.png")


# 各个模型在内部验证集上的性能指标表格
eval_test2 <- eval_test %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)
eval_test2

# 各个模型在测试集上的性能指标图示
# ROCAUC
eval_test2 %>%
  ggplot(aes(x = model, y = roc_auc, fill = model)) +
  geom_col(width = 0.5, show.legend = F) +
  geom_text(aes(label = round(roc_auc, 2)), 
            nudge_y = -0.03) +
  scale_fill_manual(values = cols4model) +
  theme_bw()
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/内部验证集ROC柱状图.png")
#############################################################

# 各个模型在内部验证集上的预测概率
predtest <- bind_rows(
  lapply(list(predtest_logistic,predtest_dt, predtest_enet,
              predtest_knn,  predtest_rf,predtest_svm, 
              predtest_xgboost, predtest_mlp,
              predtest_lightgbm), 
         "[[", 
         "prediction")
) %>%
  mutate(model = forcats::as_factor(model))
predtest

# 各个模型在测试集上的ROC
predtest %>%
  group_by(model) %>%
  roc_curve(.obs, .pred_Yes, event_level = "second") %>%
  left_join(eval_test2[, c("model", "roc_auc")]) %>%
  mutate(modelauc = paste0(model, 
                           ", ROCAUC=", round(roc_auc, 4)),
         modelauc = forcats::as_factor(modelauc)) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = modelauc)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(color = "", title = paste0("ROCs on Internal Validation Set")) +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/内部验证集ROC曲线图.png")

# 各个模型在测试集上的PR
predtest %>%
  group_by(model) %>%
  pr_curve(.obs, .pred_Yes, event_level = "second") %>%
  left_join(eval_test2[, c("model", "pr_auc")]) %>%
  mutate(modelauc = paste0(model, 
                           ", PRAUC=", round(pr_auc, 4)),
         modelauc = forcats::as_factor(modelauc)) %>%
  ggplot(aes(x = recall, y = precision, color = modelauc)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(color = "", title = paste0("PRs on Internal Validation Set")) +
  theme_bw() +
  theme(legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank())
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/内部验证集PR曲线图.png")
#############################################################

# 各个模型在测试集上的预测概率---宽数据
predtest2 <- predtest %>%
  dplyr::select(-.pred_No) %>%
  mutate(id = rep(1:nrow(predtest_logistic$prediction), 
                  length(unique(predtest$model)))) %>%
  pivot_wider(id_cols = c(id, .obs), 
              names_from = model, 
              values_from = .pred_Yes) %>%
  dplyr::select(id, .obs, sort(unique(predtest$model)))
predtest2

#############################################################


# 各个模型在测试集上的校准曲线
# 校准曲线附加置信区间
library(probably)
# 算法1
predtest %>%
  cal_plot_breaks(.obs, 
                  .pred_Yes, 
                  event_level = "second", 
                  num_breaks = 5,  # 可以改大改小
                  .by = model) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(legend.position = "none")
# 算法2
predtest %>%
  cal_plot_windowed(.obs, 
                    .pred_Yes, 
                    event_level = "second", 
                    window_size = 0.5,  # 可以改大改小
                    .by = model) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(legend.position = "none")
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/内部验证集校准曲线图.png")
# brier_score
bs <- predtest %>%
  group_by(model) %>%
  yardstick::brier_class(.obs, .pred_No) %>%
  mutate(meanpred = 0.8,
         meanobs = 0.25,
         text = paste0("BS: ", round(.estimate, 3)))
# 附加bs
predtest %>%
  cal_plot_windowed(.obs, 
                    .pred_Yes, 
                    event_level = "second", 
                    window_size = 0.5,
                    .by = model) +
  geom_text(
    bs,
    mapping = aes(x = meanpred, y = meanobs, label = text)
  ) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(legend.position = "none")
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/内部验证集BS曲线图.png")


#############################################################

# 各个模型在内部验证集上的DCA
dca_obj <- dcurves::dca(as.formula(
  paste0(".obs ~ ", 
         paste(colnames(predtest2)[3:ncol(predtest2)], 
               collapse = " + "))
),
data = predtest2,
thresholds = seq(0, 1, by = 0.01)
)
plot(dca_obj, smooth = F, span = 0.5) +
  scale_color_manual(values = c("black", "grey", cols4model)) +
  labs(title = "DCA on Internal Validation Set")
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/内部验证集DCA曲线图.png")


#######################################################################

# 各个模型在外部验证集上的性能指标
predtest_dt2$metrics
eval_testout <- bind_rows(
  lapply(list(predtest_logistic2, predtest_dt2, predtest_enet2,
              predtest_knn2, predtest_rf2,predtest_svm2, 
              predtest_xgboost2,predtest_mlp2,
              predtest_lightgbm2), 
         "[[", 
         "metrics")
) %>%
  mutate(model = forcats::as_factor(model))
eval_testout



# 平行线图
eval_testout_max <-   eval_testout %>% 
  group_by(.metric) %>%
  slice_max(.estimate)
eval_testout_min <-   eval_testout %>% 
  group_by(.metric) %>%
  slice_min(.estimate)

eval_testout %>%
  ggplot(aes(x = .metric, y = .estimate, color = model)) +
  geom_point() +
  geom_line(aes(group = model)) +
  # ggrepel::geom_text_repel(eval_max, 
  #                          mapping = aes(label = model), 
  #                          nudge_y = 0.05,
  #                          angle = 90,
  #                          show.legend = F) +
  # ggrepel::geom_text_repel(eval_min, 
  #                          mapping = aes(label = model), 
  #                          nudge_y = -0.05,
  #                          angle = 90,
  #                          show.legend = F) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/外部验证集平行线图.png")

# 指标热图
eval_testout %>%
  dplyr::select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(model = reorder(model, roc_auc)) %>%
  pivot_longer(cols = -1) %>%
  group_by(name) %>%
  mutate(valuescale = (value-min(value)) / (max(value)-min(value))) %>%
  ungroup() %>%
  ggplot(aes(x = name, y = model, fill = valuescale)) +
  geom_tile(color = "white", show.legend = F) +
  geom_text(aes(label = round(value, 2))) +
  scale_fill_gradient(low = "green", high = "red") +
  labs(x = "", y = "", fill = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, 
                                   vjust = 1,
                                   hjust = 1))
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/外部验证集热图.png")


# 各个模型在外部验证集上的性能指标表格
eval_testout2 <- eval_testout %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)
eval_testout2

# 各个模型在外部验证集上的性能指标图示
# ROCAUC
eval_testout2 %>%
  ggplot(aes(x = model, y = roc_auc, fill = model)) +
  geom_col(width = 0.5, show.legend = F) +
  geom_text(aes(label = round(roc_auc, 2)), 
            nudge_y = -0.03) +
  scale_fill_manual(values = cols4model) +
  theme_bw()
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/外部验证集ROC柱状图.png")
#############################################################

# 各个模型在外部验证集上的预测概率
predtestout <- bind_rows(
  lapply(list(predtest_logistic2,predtest_dt2, predtest_enet2,
              predtest_knn2,  predtest_rf2,predtest_svm2, 
              predtest_xgboost2, predtest_mlp2,
              predtest_lightgbm2), 
         "[[", 
         "prediction")
) %>%
  mutate(model = forcats::as_factor(model))
predtestout

# 各个模型在测试集上的ROC
predtestout %>%
  group_by(model) %>%
  roc_curve(.obs, .pred_Yes, event_level = "second") %>%
  left_join(eval_testout2[, c("model", "roc_auc")]) %>%
  mutate(modelauc = paste0(model, 
                           ", ROCAUC=", round(roc_auc, 4)),
         modelauc = forcats::as_factor(modelauc)) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = modelauc)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(color = "", title = paste0("ROCs on External Validation Set")) +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/外部验证集ROC曲线图.png")

# 各个模型在测试集上的PR
predtestout %>%
  group_by(model) %>%
  pr_curve(.obs, .pred_Yes, event_level = "second") %>%
  left_join(eval_testout2[, c("model", "pr_auc")]) %>%
  mutate(modelauc = paste0(model, 
                           ", PRAUC=", round(pr_auc, 4)),
         modelauc = forcats::as_factor(modelauc)) %>%
  ggplot(aes(x = recall, y = precision, color = modelauc)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(color = "", title = paste0("PRs on External Validation Set")) +
  theme_bw() +
  theme(legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank())
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/外部验证集PR曲线图.png")
#############################################################

# 各个模型在测试集上的预测概率---宽数据
predtestout2 <- predtestout %>%
  dplyr::select(-.pred_No) %>%
  mutate(id = rep(1:nrow(predtest_logistic2$prediction), 
                  length(unique(predtestout$model)))) %>%
  pivot_wider(id_cols = c(id, .obs), 
              names_from = model, 
              values_from = .pred_Yes) %>%
  dplyr::select(id, .obs, sort(unique(predtestout$model)))
predtestout2

#############################################################


# 各个模型在测试集上的校准曲线
# 校准曲线附加置信区间
library(probably)
# 算法1
predtestout %>%
  cal_plot_breaks(.obs, 
                  .pred_Yes, 
                  event_level = "second", 
                  num_breaks = 5,  # 可以改大改小
                  .by = model) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(legend.position = "none")
# 算法2
predtestout %>%
  cal_plot_windowed(.obs, 
                    .pred_Yes, 
                    event_level = "second", 
                    window_size = 0.5,  # 可以改大改小
                    .by = model) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(legend.position = "none")
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/外部验证集校准曲线图.png")
# brier_score
bs <- predtestout %>%
  group_by(model) %>%
  yardstick::brier_class(.obs, .pred_No) %>%
  mutate(meanpred = 0.8,
         meanobs = 0.25,
         text = paste0("BS: ", round(.estimate, 3)))
# 附加bs
predtestout %>%
  cal_plot_windowed(.obs, 
                    .pred_Yes, 
                    event_level = "second", 
                    window_size = 0.5,
                    .by = model) +
  geom_text(
    bs,
    mapping = aes(x = meanpred, y = meanobs, label = text)
  ) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(legend.position = "none")
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/外部验证集BS曲线图.png")


#############################################################

# 各个模型在内部验证集上的DCA
dca_obj <- dcurves::dca(as.formula(
  paste0(".obs ~ ", 
         paste(colnames(predtestout2)[3:ncol(predtestout2)], 
               collapse = " + "))
),
data = predtestout2,
thresholds = seq(0, 1, by = 0.01)
)
plot(dca_obj, smooth = F, span = 0.5) +
  scale_color_manual(values = c("black", "grey", cols4model)) +
  labs(title = "DCA on External Validation Set")
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/外部验证集DCA曲线图.png")



######################################################


# 自变量数据集
colnames(training_set)
traindatax <- training_set %>%
  dplyr::select(-ending)
colnames(traindatax)

# 分类型、连续型自变量名称
if(sum(sapply(traindatax, is.factor)) == 0){
  catvars <- NULL
  convars <- colnames(traindatax)
} else if(sum(sapply(traindatax, is.factor)) == ncol(traindatax)){
  catvars <- colnames(traindatax)[sapply(traindatax, is.factor)]
  convars <- NULL
} else{
  catvars <- colnames(traindatax)[sapply(traindatax, is.factor)]
  convars <- setdiff(colnames(traindatax), catvars)
}

# 提取XGBOOST最终的算法模型
final_xgboost2 <- final_xgboost %>%
  extract_fit_engine()
final_xgboost2

shapresult <- shap4cls2(
  finalmodel = final_xgboost,
  predfunc = function(model, newdata) {
    predict(model, newdata, type = "prob") %>%
      dplyr::select(ends_with(yourpositivelevel)) %>%
      pull()
  },
  datax = training_set,
  datay = training_set$ending,
  yname = "SALI incidence",
  flname = catvars,
  lxname = convars
)
# 每个样本每个变量的shap值
shapresult$shapley
# 基于shap的变量重要性
shapresult$shapvip
shapresult$shapvipplot
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/XGBoost-SHAP曲线图.png")
# 单样本预测分解
shapley <- shapviz::shapviz(
  shapresult$shapley,
  X = training_set
)
shapviz::sv_force(shapley, row_id = 1)  +  # 第1个样本
  theme(text = element_text(family = "serif"))
shapviz::sv_waterfall(shapley, row_id = 1)  +  # 第1个样本
  theme(text = element_text(family = "serif"))
# 所有分类变量的shap图示
shapresult$shapplotd_facet
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/XGBoost-SHAP分类变量图1.png")
shapresult$shapplotd_one
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/XGBoost-SHAP分类变量图2.png")
# 所有连续变量的shap图示
shapresult$shapplotc_facet
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/XGBoost-SHAP连续变量图1.png")
shapresult$shapplotc_one
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/XGBoost-SHAP连续变量图2.png")
shapresult$shapplotc_one2
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/XGBoost-SHAP连续变量图3.png")
# 所有变量一张图
# shap变量重要性
shapresult$shapvipplot_unity
# shap依赖图
shapresult$shapplot_unity

#################################################################

# shap交互作用
traindatax2 <- final_xgboost %>%
  extract_recipe() %>%
  bake(new_data = training_set) %>%
  dplyr::select(-ending)
colnames(traindatax2)
shapley2 <- shapviz::shapviz(
  final_xgboost2,
  X_pred = as.matrix(traindatax2),
  X = traindatax2,
  interactions = T
)
###################################

# 提取rf最终的算法模型
final_rf2 <- final_rf %>%
  extract_fit_engine()
final_rf2

shapresult_rf <- shap4cls2(
  finalmodel = final_rf,
  predfunc = function(model, newdata) {
    predict(model, newdata, type = "prob") %>%
      dplyr::select(ends_with(yourpositivelevel)) %>%
      pull()
  },
  datax = training_set,
  datay = training_set$ending,
  yname = "SALI incidence",
  flname = catvars,
  lxname = convars
)
# 每个样本每个变量的shap值
shapresult_rf$shapley
# 基于shap的变量重要性
shapresult_rf$shapvip
shapresult_rf$shapvipplot
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/RF-SHAP曲线图.png")
# 单样本预测分解
shapley_rf <- shapviz::shapviz(
  shapresult_rf$shapley,
  X = training_set
)
shapviz::sv_force(shapley_rf, row_id = 1)  +  # 第1个样本
  theme(text = element_text(family = "serif"))
shapviz::sv_waterfall(shapley_rf, row_id = 1)  +  # 第1个样本
  theme(text = element_text(family = "serif"))
# 所有分类变量的shap图示
shapresult_rf$shapplotd_facet
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/RF-SHAP分类变量图1.png")
shapresult_rf$shapplotd_one
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/RF-SHAP分类变量图2.png")
# 所有连续变量的shap图示
shapresult_rf$shapplotc_facet
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/RF-SHAP连续变量图1.png")
shapresult_rf$shapplotc_one
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/RF-SHAP连续变量图2.png")
shapresult_rf$shapplotc_one2
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/RF-SHAP连续变量图3.png")
# 所有变量一张图
# shap变量重要性
shapresult_rf$shapvipplot_unity
# shap依赖图
shapresult_rf$shapplot_unity
###################################

# 提取lightgbm最终的算法模型
final_lightgbm2 <- final_lightgbm %>%
  extract_fit_engine()
final_lightgbm2

shapresult_lightgbm <- shap4cls2(
  finalmodel = final_lightgbm,
  predfunc = function(model, newdata) {
    predict(model, newdata, type = "prob") %>%
      dplyr::select(ends_with(yourpositivelevel)) %>%
      pull()
  },
  datax = training_set,
  datay = training_set$ending,
  yname = "SALI incidence",
  flname = catvars,
  lxname = convars
)
# 每个样本每个变量的shap值
shapresult_lightgbm$shapley
# 基于shap的变量重要性
shapresult_lightgbm$shapvip
shapresult_lightgbm$shapvipplot
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/lightgbm-SHAP曲线图.png")
# 单样本预测分解
shapley_lightgbm <- shapviz::shapviz(
  shapresult_lightgbm$shapley,
  X = training_set
)
shapviz::sv_force(shapley_lightgbm, row_id = 1)  +  # 第1个样本
  theme(text = element_text(family = "serif"))
shapviz::sv_waterfall(shapley_lightgbm, row_id = 1)  +  # 第1个样本
  theme(text = element_text(family = "serif"))
# 所有分类变量的shap图示
shapresult_lightgbm$shapplotd_facet
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/lightgbm-SHAP分类变量图1.png")
shapresult_lightgbm$shapplotd_one
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/lightgbm-SHAP分类变量图2.png")
# 所有连续变量的shap图示
shapresult_lightgbm$shapplotc_facet
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/lightgbm-SHAP连续变量图1.png")
shapresult_lightgbm$shapplotc_one
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/lightgbm-SHAP连续变量图2.png")
shapresult_lightgbm$shapplotc_one2
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/lightgbm-SHAP连续变量图3.png")
# 所有变量一张图
# shap变量重要性
shapresult_lightgbm$shapvipplot_unity
# shap依赖图
shapresult_lightgbm$shapplot_unity

#################################################################
# 提取lightgbm最终的算法模型
final_stack2 <- final_stack %>%
  extract_fit_engine()
final_lightgbm2

shapresult_stack<- shap4cls2(
  finalmodel = final_stack,
  predfunc = function(model, newdata) {
    predict(model, newdata, type = "prob") %>%
      dplyr::select(ends_with(yourpositivelevel)) %>%
      pull()
  },
  datax = training_set,
  datay = training_set$ending,
  yname = "SALI incidence",
  flname = catvars,
  lxname = convars
)
# 每个样本每个变量的shap值
shapresult_stack$shapley
# 基于shap的变量重要性
shapresult_stack$shapvip
shapresult_stack$shapvipplot
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/stacking-SHAP曲线图.png")
# 单样本预测分解
shapley_stack <- shapviz::shapviz(
  shapresult_stack$shapley,
  X = training_set
)
shapviz::sv_force(shapley_stack, row_id = 1)  +  # 第1个样本
  theme(text = element_text(family = "serif"))
shapviz::sv_waterfall(shapley_stack, row_id = 1)  +  # 第1个样本
  theme(text = element_text(family = "serif"))
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/单样本预测分解图.png")
# 所有分类变量的shap图示
shapresult_stack$shapplotd_facet
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/stack-SHAP分类变量图1.png")
shapresult_stack$shapplotd_one
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/stack-SHAP分类变量图2.png")
# 所有连续变量的shap图示
shapresult_stack$shapplotc_facet
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/stack-SHAP连续变量图1.png")
shapresult_stack$shapplotc_one
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/stack-SHAP连续变量图2.png")
shapresult_stack$shapplotc_one2
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/stack-SHAP连续变量图3.png")




#################################################################

# shap交互作用
traindatax2 <- final_xgboost %>%
  extract_recipe() %>%
  bake(new_data = training_set) %>%
  dplyr::select(-ending)
colnames(traindatax2)
shapley2 <- shapviz::shapviz(
  final_xgboost2,
  X_pred = as.matrix(traindatax2),
  X = traindatax2,
  interactions = T
)










################################################################
#stack集成模型
load(".\\cls2\\evalresult_knn.RData")
load(".\\cls2\\evalresult_rf.RData")
load(".\\cls2\\evalresult_logistic.RData")
load(".\\cls2\\evalresult_dt.RData")
load(".\\cls2\\evalresult_enet.RData")
load(".\\cls2\\evalresult_lightgbm.RData")
load(".\\cls2\\evalresult_mlp.RData")
load(".\\cls2\\evalresult_svm.RData")
load(".\\cls2\\evalresult_xgboost.RData")
models_stack <- 
  stacks() %>% 
  add_candidates(tune_knn) %>%
  add_candidates(tune_rf) %>%
  add_candidates(tune_dt)%>%
  add_candidates(tune_enet)%>%
  add_candidates(tune_mlp)%>%
  add_candidates(tune_svm)%>%
  add_candidates(tune_xgboost)%>%
  add_candidates(tune_lightgbm)%>%
  add_candidates(cv_logistic)
models_stack

##############################

# 拟合stacking元模型——lasso
set.seed(42)
meta_stack <- blend_predictions(
  models_stack, 
  penalty = 10^seq(-2, -0.5, length = 20)
)
meta_stack
autoplot(meta_stack)
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/Stacking-lasso.png")
# 拟合选定的基础模型
set.seed(42)
final_stack <- fit_members(meta_stack)
final_stack
autoplot(final_stack)
# 应用stacking模型预测并评估

View(final_stack)
View(final_xgboost)

# 训练集
predtrain_stack <- eval4cls2(
  model = final_stack, 
  dataset = training_set, 
  yname = "ending", 
  modelname = "stacking", 
  datasetname = "training_set",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_stack$prediction
predtrain_stack$predprobplot
predtrain_stack$rocresult
predtrain_stack$rocplot
predtrain_stack$prresult
predtrain_stack$prplot
predtrain_stack$cmresult
predtrain_stack$cmplot
predtrain_stack$metrics
predtrain_stack$diycutoff
predtrain_stack$ksplot
predtrain_stack$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_stack$proc)
pROC::ci.auc(predtrain_stack$proc)

# 内部验证集
predtestin_stack <- eval4cls2(
  model = final_stack, 
  dataset = validation_set, 
  yname = "ending", 
  modelname = "stacking", 
  datasetname = "Internal Validation Set",
  cutoff = predtrain_stack$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtestin_stack$prediction
predtestin_stack$predprobplot
predtestin_stack$rocresult
predtestin_stack$rocplot
predtestin_stack$prresult
predtestin_stack$prplot
predtestin_stack$cmresult
predtestin_stack$cmplot
predtestin_stack$metrics
predtestin_stack$diycutoff
predtestin_stack$ksplot
predtestin_stack$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtest_stack$proc)
pROC::ci.auc(predtest_stack$proc)

# ROC比较检验
pROC::roc.test(predtrain_stack$proc, predtest_stack$proc)

#外部验证集
predtestout_stack <- eval4cls2(
  model = final_stack, 
  dataset = validation_set2, 
  yname = "ending", 
  modelname = "stacking", 
  datasetname = "External Validation Set",
  cutoff = predtrain_stack$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)




# 合并训练集和测试集上ROC曲线
predtrain_stack$rocresult %>%
  bind_rows(predtestin_stack$rocresult) %>%
  bind_rows(predtestout_stack$rocresult) %>%
  mutate(dataAUC = paste(data, " ROCAUC:", round(ROCAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/Stacking模型ROC曲线.png")


# 合并训练集和测试集上PR曲线
predtrain_stack$prresult %>%
  bind_rows(predtestin_stack$prresult) %>%
  bind_rows(predtestout_stack$prresult) %>%
  mutate(dataAUC = paste(data, " PRAUC:", round(PRAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank())
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/Stacking模型PR曲线.png")








# 合并训练集和测试集上性能指标
predtrain_stack$metrics %>%
  bind_rows(predtestin_stack$metrics) %>%
  bind_rows(predtestout_stack$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 保存评估结果
save(predtrain_stack,
     predtestin_stack,
     predtestout_stack,
     file = ".\\cls2\\evalresult_stack.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_stack_heart <- final_stack
traindata_heart <- training_set
save(final_stack_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_stack_heart.RData")


# 各个模型在测试集上的性能指标
predtrain_dt$metrics
eval_train_all <- bind_rows(
  lapply(list(predtrain_logistic, predtrain_dt, predtrain_enet,
              predtrain_knn, predtrain_rf,predtrain_svm, 
              predtrain_xgboost,predtrain_mlp,
              predtrain_lightgbm,predtrain_stack), 
         "[[", 
         "metrics")
) %>%
  mutate(model = forcats::as_factor(model))
eval_train_all



# 平行线图
eval_train_all_max <-   eval_train_all %>% 
  group_by(.metric) %>%
  slice_max(.estimate)
eval_train_all_min <-   eval_train_all %>% 
  group_by(.metric) %>%
  slice_min(.estimate)

eval_train_all %>%
  ggplot(aes(x = .metric, y = .estimate, color = model)) +
  geom_point() +
  geom_line(aes(group = model)) +
  # ggrepel::geom_text_repel(eval_max, 
  #                          mapping = aes(label = model), 
  #                          nudge_y = 0.05,
  #                          angle = 90,
  #                          show.legend = F) +
  # ggrepel::geom_text_repel(eval_min, 
  #                          mapping = aes(label = model), 
  #                          nudge_y = -0.05,
  #                          angle = 90,
  #                          show.legend = F) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/训练集all平行线图.png")

# 指标热图
eval_train_all %>%
  dplyr::select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(model = reorder(model, roc_auc)) %>%
  pivot_longer(cols = -1) %>%
  group_by(name) %>%
  mutate(valuescale = (value-min(value)) / (max(value)-min(value))) %>%
  ungroup() %>%
  ggplot(aes(x = name, y = model, fill = valuescale)) +
  geom_tile(color = "white", show.legend = F) +
  geom_text(aes(label = round(value, 2))) +
  scale_fill_gradient(low = "green", high = "red") +
  labs(x = "", y = "", fill = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, 
                                   vjust = 1,
                                   hjust = 1))
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/训练集all热图.png")

# 各个模型在测试集上的性能指标

eval_testin_all <- bind_rows(
  lapply(list(predtest_logistic, predtest_dt, predtest_enet,
              predtest_knn, predtest_rf,predtest_svm, 
              predtest_xgboost,predtest_mlp,
              predtest_lightgbm,predtestin_stack), 
         "[[", 
         "metrics")
) %>%
  mutate(model = forcats::as_factor(model))
eval_testin_all



# 平行线图
eval_testin_max <-   eval_testin_all %>% 
  group_by(.metric) %>%
  slice_max(.estimate)
eval_testin_min <-   eval_testin_all %>% 
  group_by(.metric) %>%
  slice_min(.estimate)

eval_testin_all %>%
  ggplot(aes(x = .metric, y = .estimate, color = model)) +
  geom_point() +
  geom_line(aes(group = model)) +
  # ggrepel::geom_text_repel(eval_max, 
  #                          mapping = aes(label = model), 
  #                          nudge_y = 0.05,
  #                          angle = 90,
  #                          show.legend = F) +
  # ggrepel::geom_text_repel(eval_min, 
  #                          mapping = aes(label = model), 
  #                          nudge_y = -0.05,
  #                          angle = 90,
  #                          show.legend = F) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/内部验证all平行线图.png")

# 指标热图
eval_testin_all %>%
  dplyr::select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(model = reorder(model, roc_auc)) %>%
  pivot_longer(cols = -1) %>%
  group_by(name) %>%
  mutate(valuescale = (value-min(value)) / (max(value)-min(value))) %>%
  ungroup() %>%
  ggplot(aes(x = name, y = model, fill = valuescale)) +
  geom_tile(color = "white", show.legend = F) +
  geom_text(aes(label = round(value, 2))) +
  scale_fill_gradient(low = "green", high = "red") +
  labs(x = "", y = "", fill = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, 
                                   vjust = 1,
                                   hjust = 1))
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/内部验证集all热图.png")

# 各个模型在测试集上的性能指标
eval_testout_all <- bind_rows(
  lapply(list(predtest_logistic2, predtest_dt2, predtest_enet2,
              predtest_knn2, predtest_rf2,predtest_svm2, 
              predtest_xgboost2,predtest_mlp2,
              predtest_lightgbm2,predtestout_stack), 
         "[[", 
         "metrics")
) %>%
  mutate(model = forcats::as_factor(model))
eval_testout_all



# 平行线图
eval_testout_max <-   eval_testout_all %>% 
  group_by(.metric) %>%
  slice_max(.estimate)
eval_testout_min <-   eval_testout_all %>% 
  group_by(.metric) %>%
  slice_min(.estimate)

eval_testout_all %>%
  ggplot(aes(x = .metric, y = .estimate, color = model)) +
  geom_point() +
  geom_line(aes(group = model)) +
  # ggrepel::geom_text_repel(eval_max, 
  #                          mapping = aes(label = model), 
  #                          nudge_y = 0.05,
  #                          angle = 90,
  #                          show.legend = F) +
  # ggrepel::geom_text_repel(eval_min, 
  #                          mapping = aes(label = model), 
  #                          nudge_y = -0.05,
  #                          angle = 90,
  #                          show.legend = F) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/外部验证集all平行线图.png")

# 指标热图
eval_testout_all %>%
  dplyr::select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(model = reorder(model, precision)) %>%
  pivot_longer(cols = -1) %>%
  group_by(name) %>%
  mutate(valuescale = (value-min(value)) / (max(value)-min(value))) %>%
  ungroup() %>%
  ggplot(aes(x = name, y = model, fill = valuescale)) +
  geom_tile(color = "white", show.legend = F) +
  geom_text(aes(label = round(value, 2))) +
  scale_fill_gradient(low = "green", high = "red") +
  labs(x = "", y = "", fill = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, 
                                   vjust = 1,
                                   hjust = 1))
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/外部验证集all热图.png")

eval_testout_all %>%
  dplyr::select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(model = factor(model, levels = c("stacking", setdiff(unique(model), "stacking")))) %>%
  arrange(match(model, levels(model))) %>%
  pivot_longer(cols = -1) %>%
  group_by(name) %>%
  mutate(valuescale = (value - min(value)) / (max(value) - min(value))) %>%
  ungroup() %>%
  ggplot(aes(x = name, y = model, fill = valuescale)) +
  geom_tile(color = "white", show.legend = F) +
  geom_text(aes(label = round(value, 2))) +
  scale_fill_gradient(low = "green", high = "red") +
  labs(x = "", y = "", fill = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, 
                                   vjust = 1,
                                   hjust = 1))


# 各个模型交叉验证的各折指标点线图
evalcv <- bind_rows(
  lapply(list(evalcv_logistic, evalcv_dt, evalcv_enet,
              evalcv_knn, evalcv_svm, evalcv_rf,
              evalcv_xgboost, evalcv_mlp, evalcv_lightgbm), 
         "[[", 
         "evalcv")
) %>%
  mutate(
    model = forcats::as_factor(model),
    modelperf = paste0(model, "(", round(mean, 2),"±",
                       round(sd,2), ")")
  )
evalcv

# ROC
evalcvroc_max <-   evalcv %>% 
  filter(.metric == "roc_auc") %>%
  group_by(id) %>%
  slice_max(.estimate)
evalcvroc_min <-   evalcv %>% 
  filter(.metric == "roc_auc") %>%
  group_by(id) %>%
  slice_min(.estimate)
evalcv %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = id, y = .estimate, 
             group = modelperf, color = modelperf)) +
  geom_point() +
  geom_line() +
  ggrepel::geom_text_repel(evalcvroc_max, 
                           mapping = aes(label = model), 
                           nudge_y = 0.01,
                           show.legend = F) +
  ggrepel::geom_text_repel(evalcvroc_min, 
                           mapping = aes(label = model), 
                           nudge_y = -0.01,
                           show.legend = F) +
  # scale_y_continuous(limits = c(0.55, 0.85)) +
  scale_color_manual(values = cols4model) +
  labs(x = "", y = "ROCAUC", color = "Model") +
  theme_bw()
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/训练集折线图.png")
# PR
evalcvpr_max <-   evalcv %>% 
  filter(.metric == "pr_auc") %>%
  group_by(id) %>%
  slice_max(.estimate)
evalcvpr_min <-   evalcv %>% 
  filter(.metric == "pr_auc") %>%
  group_by(id) %>%
  slice_min(.estimate)
evalcv %>%
  filter(.metric == "pr_auc") %>%
  ggplot(aes(x = id, y = .estimate, 
             group = modelperf, color = modelperf)) +
  geom_point() +
  geom_line() +
  ggrepel::geom_text_repel(evalcvpr_max, 
                           mapping = aes(label = model), 
                           nudge_y = 0.01,
                           show.legend = F) +
  ggrepel::geom_text_repel(evalcvpr_min, 
                           mapping = aes(label = model), 
                           nudge_y = -0.01,
                           show.legend = F) +
  # scale_y_continuous(limits = c(0.5, 0.9)) +
  scale_color_manual(values = cols4model) +
  labs(x = "", y = "prAUC", color = "Model") +
  theme_bw()
#懂又不懂，上面限制了y的范围，直接注释掉
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/训练集PR折线图.png")

# 各个模型交叉验证的指标平均值图(带上下限)
# ROC
evalcv %>%
  filter(.metric == "roc_auc") %>%
  group_by(model) %>%
  sample_n(size = 1) %>%
  ungroup() %>%
  ggplot(aes(x = model, y = mean, color = model)) +
  geom_point(size = 2, show.legend = F) +
  # geom_line(group = 1) +
  geom_errorbar(aes(ymin = mean-sd, 
                    ymax = mean+sd),
                width = 0.1, 
                linewidth = 1.2,
                show.legend = F) +
  # scale_y_continuous(limits = c(0.7, 1)) +
  scale_color_manual(values = cols4model) +
  labs(y = "cv roc_auc") +
  theme_bw()
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/训练集交叉验证指标图.png")
# PR
evalcv %>%
  filter(.metric == "pr_auc") %>%
  group_by(model) %>%
  sample_n(size = 1) %>%
  ungroup() %>%
  ggplot(aes(x = model, y = mean, color = model)) +
  geom_point(size = 2, show.legend = F) +
  # geom_line(group = 1) +
  geom_errorbar(aes(ymin = mean-sd, 
                    ymax = mean+sd),
                width = 0.1, 
                linewidth = 1.2,
                show.legend = F) +
  # scale_y_continuous(limits = c(0.7, 1)) +
  scale_color_manual(values = cols4model) +
  labs(y = "cv pr_auc") +
  theme_bw()
ggsave(file="/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/训练集交叉验证PR指标图.png")




# 各个模型在测试集上的预测概率
predtrain <- bind_rows(
  lapply(list(predtrain_logistic, predtrain_dt, predtrain_enet,
              predtrain_knn,  predtrain_rf,predtrain_svm, 
              predtrain_xgboost, predtrain_mlp,
              predtrain_lightgbm), 
         "[[", 
         "prediction")
) %>%
  mutate(model = forcats::as_factor(model))
predtrain


# 测试集ROC曲线
eval_train <- bind_rows(
  lapply(list(predtrain_logistic, predtrain_dt, predtrain_enet,
              predtrain_knn, predtrain_rf,predtrain_svm, 
              predtrain_xgboost,predtrain_lightgbm,
              predtrain_mlp), 
         "[[", 
         "metrics")
) %>%
  mutate(model = forcats::as_factor(model))
eval_t
eval_train

eval_train2 <- eval_train %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 合并测试集上ROC曲线
predtrain %>%
  group_by(model) %>%
  roc_curve(.obs, .pred_Yes, event_level = "second") %>%
  left_join(eval_train2[, c("model", "roc_auc")]) %>%
  mutate(modelauc = paste0(model, 
                           ", ROCAUC=", round(roc_auc, 4)),
         modelauc = forcats::as_factor(modelauc)) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = modelauc)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(color = "", title = paste0("ROCs on training_set")) +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())
ggsave("/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/roc_train.png")


# 各个模型在训练集上的PR
predtrain %>%
  group_by(model) %>%
  pr_curve(.obs, .pred_Yes, event_level = "second") %>%
  left_join(eval_train2[, c("model", "pr_auc")]) %>%
  mutate(modelauc = paste0(model, 
                           ", PRAUC=", round(pr_auc, 4)),
         modelauc = forcats::as_factor(modelauc)) %>%
  ggplot(aes(x = recall, y = precision, color = modelauc)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(color = "", title = paste0("PRs on training_set")) +
  theme_bw() +
  theme(legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank())
ggsave("/Users/xy3yy_qj/Desktop/qi_jing/脓毒症肝损伤and机器学习/pr_train.png")




#stack集成模型
load(".\\cls2\\evalresult_knn.RData")
load(".\\cls2\\evalresult_rf.RData")
load(".\\cls2\\evalresult_logistic.RData")
load(".\\cls2\\evalresult_dt.RData")
load(".\\cls2\\evalresult_enet.RData")
load(".\\cls2\\evalresult_lightgbm.RData")
load(".\\cls2\\evalresult_mlp.RData")
load(".\\cls2\\evalresult_svm.RData")
load(".\\cls2\\evalresult_xgboost.RData")
models_stack <- 
  stacks() %>% 
  add_candidates(tune_knn) %>%
  add_candidates(tune_rf) %>%
  add_candidates(tune_dt)%>%
  add_candidates(tune_enet)%>%
  add_candidates(tune_mlp)%>%
  add_candidates(tune_svm)%>%
  add_candidates(tune_xgboost)%>%
  add_candidates(cv_logistic)
models_stack

##############################

# 拟合stacking元模型——lasso
set.seed(42)
meta_stack <- blend_predictions(
  models_stack, 
  penalty = 10^seq(-2, -0.5, length = 20)
)
meta_stack
autoplot(meta_stack)

# 拟合选定的基础模型
set.seed(42)
final_stack <- fit_members(meta_stack)
final_stack

# 应用stacking模型预测并评估

# 训练集
predtrain_stack <- eval4cls2(
  model = final_stack, 
  dataset = training_set, 
  yname = "ending", 
  modelname = "stacking", 
  datasetname = "training_set",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_stack$prediction
predtrain_stack$predprobplot
predtrain_stack$rocresult
predtrain_stack$rocplot
predtrain_stack$prresult
predtrain_stack$prplot
predtrain_stack$cmresult
predtrain_stack$cmplot
predtrain_stack$metrics
predtrain_stack$diycutoff
predtrain_stack$ksplot
predtrain_stack$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_stack$proc)
pROC::ci.auc(predtrain_stack$proc)

# 测试集
predtest_stack <- eval4cls2(
  model = final_stack, 
  dataset = validation_set, 
  yname = "ending", 
  modelname = "stacking", 
  datasetname = "validation_set",
  cutoff = predtrain_stack$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_stack$prediction
predtest_stack$predprobplot
predtest_stack$rocresult
predtest_stack$rocplot
predtest_stack$prresult
predtest_stack$prplot
predtest_stack$cmresult
predtest_stack$cmplot
predtest_stack$metrics
predtest_stack$diycutoff
predtest_stack$ksplot
predtest_stack$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtest_stack$proc)
pROC::ci.auc(predtest_stack$proc)

# ROC比较检验
pROC::roc.test(predtrain_stack$proc, predtest_stack$proc)

# 合并训练集和测试集上ROC曲线
predtrain_stack$rocresult %>%
  bind_rows(predtest_stack$rocresult) %>%
  mutate(dataAUC = paste(data, " ROCAUC:", round(ROCAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上PR曲线
predtrain_stack$prresult %>%
  bind_rows(predtest_stack$prresult) %>%
  mutate(dataAUC = paste(data, " PRAUC:", round(PRAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上性能指标
final_table <- predtrain_stack$metrics %>%
  bind_rows(predtestin_stack$metrics) %>%
  bind_rows(predtestout_stack$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

final_table_fit <- final_table %>%
  mutate(F1 = (2*precision*recall)/(precision + recall))


View(predtrain_stack$metrics)
# 保存评估结果
save(predtrain_stack,
     predtest_stack,
     file = ".\\cls2\\evalresult_stack.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_stack_heart <- final_stack
traindata_heart <- training_set
save(final_stack_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_stack_heart.RData")


# 模型机器---R语言tidymodels包机器学习分类与回归模型---二分类---模型比较

#############################################################
# remotes::install_github("tidymodels/probably")
library(tidymodels)

# 加载各个模型的评估结果
evalfiles <- list.files(".\\cls2\\", full.names = T)
lapply(evalfiles, load, .GlobalEnv)

# 横向比较的模型个数
nmodels <- 10
cols4model <- rainbow(nmodels)  # 模型统一配色
#############################################################

# 各个模型在测试集上的性能指标
predtest_dt$metrics
eval <- bind_rows(
  lapply(list(predtest_logistic, predtest_dt, predtest_enet,
              predtest_knn, predtest_rf,predtest_svm, 
              predtest_xgboost,predtest_mlp,
              predtest_stack), 
         "[[", 
         "metrics")
) %>%
  mutate(model = forcats::as_factor(model))
eval



# 平行线图
eval_max <-   eval %>% 
  group_by(.metric) %>%
  slice_max(.estimate)
eval_min <-   eval %>% 
  group_by(.metric) %>%
  slice_min(.estimate)

eval %>%
  ggplot(aes(x = .metric, y = .estimate, color = model)) +
  geom_point() +
  geom_line(aes(group = model)) +
  # ggrepel::geom_text_repel(eval_max, 
  #                          mapping = aes(label = model), 
  #                          nudge_y = 0.05,
  #                          angle = 90,
  #                          show.legend = F) +
  # ggrepel::geom_text_repel(eval_min, 
  #                          mapping = aes(label = model), 
  #                          nudge_y = -0.05,
  #                          angle = 90,
  #                          show.legend = F) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# 指标热图
eval %>%
  dplyr::select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(model = reorder(model, roc_auc)) %>%
  pivot_longer(cols = -1) %>%
  group_by(name) %>%
  mutate(valuescale = (value-min(value)) / (max(value)-min(value))) %>%
  ungroup() %>%
  ggplot(aes(x = name, y = model, fill = valuescale)) +
  geom_tile(color = "white", show.legend = F) +
  geom_text(aes(label = round(value, 2))) +
  scale_fill_gradient(low = "green", high = "red") +
  labs(x = "", y = "", fill = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, 
                                   vjust = 1,
                                   hjust = 1))

# 各个模型在测试集上的性能指标表格
eval2 <- eval %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)
eval2

# 各个模型在测试集上的性能指标图示
# ROCAUC
eval2 %>%
  ggplot(aes(x = model, y = roc_auc, fill = model)) +
  geom_col(width = 0.5, show.legend = F) +
  geom_text(aes(label = round(roc_auc, 2)), 
            nudge_y = -0.03) +
  scale_fill_manual(values = cols4model) +
  theme_bw()

#############################################################

# 各个模型在测试集上的预测概率
predtest <- bind_rows(
  lapply(list(predtest_logistic, predtest_dt, predtest_enet,
              predtest_knn,  predtest_rf,predtest_svm, 
              predtest_xgboost, predtest_mlp,
              predtest_stack), 
         "[[", 
         "prediction")
) %>%
  mutate(model = forcats::as_factor(model))
predtest

# 各个模型在测试集上的ROC
predtest %>%
  group_by(model) %>%
  roc_curve(.obs, .pred_Yes, event_level = "second") %>%
  left_join(eval2[, c("model", "roc_auc")]) %>%
  mutate(modelauc = paste0(model, 
                           ", ROCAUC=", round(roc_auc, 4)),
         modelauc = forcats::as_factor(modelauc)) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = modelauc)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(color = "", title = paste0("ROCs on testdata")) +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 各个模型在测试集上的PR
predtest %>%
  group_by(model) %>%
  pr_curve(.obs, .pred_Yes, event_level = "second") %>%
  left_join(eval2[, c("model", "pr_auc")]) %>%
  mutate(modelauc = paste0(model, 
                           ", PRAUC=", round(pr_auc, 4)),
         modelauc = forcats::as_factor(modelauc)) %>%
  ggplot(aes(x = recall, y = precision, color = modelauc)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(color = "", title = paste0("PRs on testdata")) +
  theme_bw() +
  theme(legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank())

#############################################################

# 各个模型在测试集上的预测概率---宽数据
predtest2 <- predtest %>%
  dplyr::select(-.pred_No) %>%
  mutate(id = rep(1:nrow(predtest_logistic$prediction), 
                  length(unique(predtest$model)))) %>%
  pivot_wider(id_cols = c(id, .obs), 
              names_from = model, 
              values_from = .pred_Yes) %>%
  dplyr::select(id, .obs, sort(unique(predtest$model)))
predtest2

#############################################################


# 各个模型在测试集上的校准曲线
# 校准曲线附加置信区间
library(probably)
# 算法1
predtest %>%
  cal_plot_breaks(.obs, 
                  .pred_Yes, 
                  event_level = "second", 
                  num_breaks = 5,  # 可以改大改小
                  .by = model) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(legend.position = "none")
# 算法2
predtest %>%
  cal_plot_windowed(.obs, 
                    .pred_Yes, 
                    event_level = "second", 
                    window_size = 0.5,  # 可以改大改小
                    .by = model) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(legend.position = "none")

# brier_score
bs <- predtest %>%
  group_by(model) %>%
  yardstick::brier_class(.obs, .pred_No) %>%
  mutate(meanpred = 0.8,
         meanobs = 0.25,
         text = paste0("BS: ", round(.estimate, 3)))
# 附加bs
predtest %>%
  cal_plot_windowed(.obs, 
                    .pred_Yes, 
                    event_level = "second", 
                    window_size = 0.5,
                    .by = model) +
  geom_text(
    bs,
    mapping = aes(x = meanpred, y = meanobs, label = text)
  ) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(legend.position = "none")



#############################################################

# 各个模型在测试集上的DCA
dca_obj <- dcurves::dca(as.formula(
  paste0(".obs ~ ", 
         paste(colnames(predtest2)[3:ncol(predtest2)], 
               collapse = " + "))
),
data = predtest2,
thresholds = seq(0, 1, by = 0.01)
)
plot(dca_obj, smooth = T, span = 0.5) +
  scale_color_manual(values = c("black", "grey", cols4model)) +
  labs(title = "DCA on testdata")

#############################################################

# 各个模型交叉验证的各折指标点线图
evalcv <- bind_rows(
  lapply(list(evalcv_logistic, evalcv_dt, evalcv_enet,
              evalcv_knn, evalcv_svm, evalcv_rf,
              evalcv_xgboost, evalcv_mlp), 
         "[[", 
         "evalcv")
) %>%
  mutate(
    model = forcats::as_factor(model),
    modelperf = paste0(model, "(", round(mean, 2),"±",
                       round(sd,2), ")")
  )
evalcv

# ROC
evalcvroc_max <-   evalcv %>% 
  filter(.metric == "roc_auc") %>%
  group_by(id) %>%
  slice_max(.estimate)
evalcvroc_min <-   evalcv %>% 
  filter(.metric == "roc_auc") %>%
  group_by(id) %>%
  slice_min(.estimate)
evalcv %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = id, y = .estimate, 
             group = modelperf, color = modelperf)) +
  geom_point() +
  geom_line() +
  ggrepel::geom_text_repel(evalcvroc_max, 
                           mapping = aes(label = model), 
                           nudge_y = 0.01,
                           show.legend = F) +
  ggrepel::geom_text_repel(evalcvroc_min, 
                           mapping = aes(label = model), 
                           nudge_y = -0.01,
                           show.legend = F) +
  # scale_y_continuous(limits = c(0.55, 0.85)) +
  scale_color_manual(values = cols4model) +
  labs(x = "", y = "ROCAUC", color = "Model") +
  theme_bw()

# PR
evalcvpr_max <-   evalcv %>% 
  filter(.metric == "pr_auc") %>%
  group_by(id) %>%
  slice_max(.estimate)
evalcvpr_min <-   evalcv %>% 
  filter(.metric == "pr_auc") %>%
  group_by(id) %>%
  slice_min(.estimate)
evalcv %>%
  filter(.metric == "pr_auc") %>%
  ggplot(aes(x = id, y = .estimate, 
             group = modelperf, color = modelperf)) +
  geom_point() +
  geom_line() +
  ggrepel::geom_text_repel(evalcvpr_max, 
                           mapping = aes(label = model), 
                           nudge_y = 0.01,
                           show.legend = F) +
  ggrepel::geom_text_repel(evalcvpr_min, 
                           mapping = aes(label = model), 
                           nudge_y = -0.01,
                           show.legend = F) +
  # scale_y_continuous(limits = c(0.5, 0.9)) +
  scale_color_manual(values = cols4model) +
  labs(x = "", y = "prAUC", color = "Model") +
  theme_bw()
#懂又不懂，上面限制了y的范围，直接注释掉


# 各个模型交叉验证的指标平均值图(带上下限)
# ROC
evalcv %>%
  filter(.metric == "roc_auc") %>%
  group_by(model) %>%
  sample_n(size = 1) %>%
  ungroup() %>%
  ggplot(aes(x = model, y = mean, color = model)) +
  geom_point(size = 2, show.legend = F) +
  # geom_line(group = 1) +
  geom_errorbar(aes(ymin = mean-sd, 
                    ymax = mean+sd),
                width = 0.1, 
                linewidth = 1.2,
                show.legend = F) +
  # scale_y_continuous(limits = c(0.7, 1)) +
  scale_color_manual(values = cols4model) +
  labs(y = "cv roc_auc") +
  theme_bw()

# PR
evalcv %>%
  filter(.metric == "pr_auc") %>%
  group_by(model) %>%
  sample_n(size = 1) %>%
  ungroup() %>%
  ggplot(aes(x = model, y = mean, color = model)) +
  geom_point(size = 2, show.legend = F) +
  # geom_line(group = 1) +
  geom_errorbar(aes(ymin = mean-sd, 
                    ymax = mean+sd),
                width = 0.1, 
                linewidth = 1.2,
                show.legend = F) +
  # scale_y_continuous(limits = c(0.7, 1)) +
  scale_color_manual(values = cols4model) +
  labs(y = "cv pr_auc") +
  theme_bw()