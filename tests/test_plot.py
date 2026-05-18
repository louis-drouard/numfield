import pytest
import matplotlib.pyplot as plt

# Test plotting (using pytest-mpl)
# Use pytest-mpl to check the plot matches a reference image

@pytest.mark.mpl_image_compare(baseline_dir='baseline', tolerance=15)
def test_plotting_field(coarse_field):
    fig, ax = coarse_field.plot(2,init_slice=0)   
    assert fig is not None
    assert ax is not None
    return fig

@pytest.mark.mpl_image_compare(baseline_dir='baseline', tolerance=15)
def test_projection_plot(coarse_field, fine_mesh):
    fig, ax = coarse_field.project_on(fine_mesh).plot(2,init_slice=0)   
    assert fig is not None
    assert ax is not None
    return fig

@pytest.mark.mpl_image_compare(baseline_dir='baseline', tolerance=15)
def test_plot_x_axis(fine_field):

    fig, ax = fine_field.plot(1,init_slice=0)   
    assert fig is not None
    assert ax is not None

    return fig

@pytest.mark.mpl_image_compare(baseline_dir='baseline', tolerance=15)
def test_plot_y_axis(fine_field):

    fig, ax = fine_field.plot(0,init_slice=0)   
    assert fig is not None
    assert ax is not None

    return fig

@pytest.mark.mpl_image_compare(baseline_dir='baseline', tolerance=15)
def test_plot_1d(fine_field):

    fig, ax = fine_field[:,0,0].plot()   
    assert fig is not None
    assert ax is not None

    return fig