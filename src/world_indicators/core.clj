(ns world-indicators.core
  (:require [clojure.java.io :as io]
            [clojure.string :as string]
            [clojure.data.csv :as csv]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [cortex.optimize.adam :as adam]
            [cortex.metrics :as metrics]
            [cortex.util :as util]
            [cortex.experiment.train :as experiment-train]))

(def data-file "data/knn_imputed_indicators.csv") ;7253

(def params
  {:test-size   1920
   :optimizer   (adam/adam)
   :batch-size  128
   :epoch-size  1024})

(def dataset
  (future
    (let [ind-data (with-open [infile (io/reader data-file)]
                          (rest (doall (csv/read-csv infile))))
          data (->> ind-data
                    (map rest)                ; drop first col (label)
                    (map #(map read-string %)))
          labels (->> ind-data
                      (map first)
                      (map read-string))]
        (mapv (fn [d l] {:data d :label l}) data labels))))

(defn infinite-dataset
  "Given a finite dataset, generate an infinite sequence of maps partitioned
  by :epoch-size"
  [map-seq & {:keys [epoch-size]
              :or {epoch-size 1024}}]
  (->> (repeatedly #(shuffle map-seq))
       (mapcat identity)
       (partition epoch-size)))


(def network-description
  [(layers/input (count (:data (first @dataset))) 1 1 :id :data)
   (layers/linear->relu 8)
   (layers/linear 1 :id :label)])


(defn train
 "Trains network for :epoch-count number of epochs"
 []
 (let [network (network/linear-network network-description)
       [train-ds test-ds] [(infinite-dataset (drop (:test-size params) @dataset))
                           (take (:test-size params) @dataset)]]
   (experiment-train/train-n network train-ds test-ds
                                     :batch-size (:batch-size params)
                                     :epoch-count (:epoch-count params)
                                     :optimizer (:optimizer params))))
