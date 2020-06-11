# Matlab code description :
Note that ``data_generation``, ``data_test_generation`` , ``data_test_circle`` , ``prediction_vizualisation`` are templates. You need to change the path , order , mesh and discontinuity in order to reproduce a dataset.
- ***Data generation :***
    - ``data_gen`` : Compute and collecte data for a single triangle in the reference element.
    - ``data_gen_close_neighbor`` : Compute and collecte data for a triangle and the egdes of its neighbor in the reference element.
    - ``data_gen_all_neighborhood`` : Compute and collecte data for a triangle and the triangle of its neighbor in the reference element.
    - ``data_gen_all_neighborhood_orthogonal`` : Compute and collecte data for the location of the discontinuity for a triangle and the triangle of its neighbor in the reference element.
    - ``data_generation`` : Template of main function that allows to compute the data for different discontinuities, orders, meshes and information of the troubled-cells.
- ***Data test generation :***
    - ``data_test_generation`` : Template of the main function that allows to compute test data for different discontinuities (30 ,60 ,90 , circular), orders, meshes and information of the troubled-cells.
    - ``data_test_circle`` : Template of the main function that allows to compute test data for different discontinuities (30 ,60 ,90 , circular), orders, meshes and information of the troubled-cells.
- ***Vizualisation :***
    - ``pred_plot`` : Plot the prediction of the orientation of a discontinuity.
    - ``pred_plot_ortho`` : Plot the prediction of the orientation and location of a discontinuity.
    - ``prediction_vizualisation`` : Template of the main function that allows to change the orders, meshes and predictions.
    
- ***Other function :***
    - ``transformation_phy_to_ref`` : Transform a point in the physical space to the reference space.
    - ``transformation_ref_to_phy`` : Transform a point in the reference space to the physical space.
    - ``playground_location`` : Test to find the intersection and orthogonal vector of a discontinuity.