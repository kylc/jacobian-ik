(ns ik.core
  (:require
   [clojure.core.matrix :as np]
   [quil.core :as q]
   [quil.middleware :as m]))

(np/set-current-implementation :vectorz)

(def p-gain 0.00002)
(def initial-state
  {:link-lengths [100.0 100.0]
   :joint-angles [(/ Math/PI -4.0) (/ Math/PI 4.0)]})

(defn joint-tform
  "Compute a transformation matrix which transforms a point from the origin to
  the joint's end-effector, for a joint with length `L` and angle `q`."
  [L q]
  (np/matrix [[(Math/cos q) (- (Math/sin q)) 0 (* L (Math/cos q))]
              [(Math/sin q) (Math/cos q) 0 (* L (Math/sin q))]
              [0.0 0.0 1.0 0.0]
              [0.0 0.0 0.0 1.0]]))

(defn solve-fk
  "Compute the transforms of each link in the serial chain defined by `Ls` with
  joint angles `qs`, and the transform of the end-effector."
  [Ls qs]
  (->>
   (map vector Ls qs)
   (reductions
    (fn [tfm [L q]]
      (np/mmul tfm (joint-tform L q)))
    (np/identity-matrix 4 4))))

(defn solve-fk-last
  "Compute the transform of the end-effector of a serial joint chain."
  [Ls qs]
  (-> (solve-fk Ls qs) last))

(defn solve-fk-pt
  [Ls qs]
  (let [origin (np/column-matrix [0.0 0.0 0.0 1.0])
        A (solve-fk-last Ls qs)]
    (np/mmul A origin)))

(defn jac-col
  "Compute a single column in the Jacobian matrix (linear and angular components
  for a single joint) for joint index `n`."
  [Ls qs n dq]
  (let [origin (np/column-matrix [0.0 0.0 0.0 1.0])
        a (solve-fk-pt Ls (update qs n + dq))
        b (solve-fk-pt Ls qs)]
    (->
     (np/sub a b)
     (np/scale (/ 1.0 dq))
     (np/submatrix 0 3 0 1))))

(defn jac [Ls qs dq]
  (->>
   (for [i (range (count Ls))]
     (jac-col Ls qs i dq))
   (apply np/join-along 1)))

(defn setup []
  (q/frame-rate 60)
  (q/color-mode :rgb)
  initial-state)

(defn update-state [{:keys [link-lengths joint-angles] :as state}]
  (let [;; Compute the Jacobian inverse for the current robot configuration.
        J (time (jac link-lengths joint-angles 0.01))
        J_T (np/transpose J)

        ;; Compute a position error term.
        zero-pt (np/column-matrix [0.0 0.0 0.0 1.0])
        current-pos (-> (solve-fk-last link-lengths joint-angles)
                        (np/mmul zero-pt))
        target-pos (np/column-matrix [(- (q/mouse-x) (/ (q/width) 2)) (- (q/mouse-y) (/ (q/height) 2)) 0.0 1.0])
        pos-error (-> (np/sub target-pos current-pos)
                      (np/submatrix 0 3 0 1))

        ;; Compute a feedback term and compute joint velocities using the Jacobian inverse.
        desired-velocity (np/scale pos-error p-gain)
        joint-vel (np/mmul J_T desired-velocity)]
    (-> state
        (update-in [:joint-angles 0] + (np/mget joint-vel 0 0))
        (update-in [:joint-angles 1] + (np/mget joint-vel 1 0)))))

(defn draw-state [{:keys [link-lengths joint-angles]}]
  (q/background 30 30 30)

  (q/with-translation [(/ (q/width) 2) (/ (q/height) 2)]
    (let [joint-tfms (solve-fk link-lengths joint-angles)]
      ;;  Lines connecting the joints.
      (doseq [[a b] (partition 2 1 joint-tfms)]
        (let [zero-pt (np/column-matrix [0.0 0.0 0.0 1.0])
              a-pos (np/get-column (np/mmul a zero-pt) 0)
              b-pos (np/get-column (np/mmul b zero-pt) 0)]
          (q/stroke 255 255 0)
          (q/line (take 2 a-pos) (take 2 b-pos))))

      ;; Circles representing the joint origins.
      (doseq [a joint-tfms]
        (let [zero-pt (np/column-matrix [0.0 0.0 0.0 1.0])
              a-pos (np/get-column (np/mmul a zero-pt) 0)]
          (q/fill 255 0 0)
          (q/ellipse (np/mget a-pos 0) (np/mget a-pos 1) 10 10))))))

(q/defsketch ik
  :title "IK Demo"
  :size [500 500]
  :setup setup
  :update update-state
  :draw draw-state
  :middleware [m/fun-mode])

(defn -main [])
