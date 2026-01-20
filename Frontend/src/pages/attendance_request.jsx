import React, { useState, useEffect } from 'react';
import { Clock, CheckCircle, XCircle, AlertCircle, User, Calendar, Filter, X, Search, RefreshCw, FileText, Shield, Building2 } from 'lucide-react';

const AttendanceRequestPage = () => {
  const [requests, setRequests] = useState([]);
  const [employees, setEmployees] = useState({});
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('pending');
  const [approving, setApproving] = useState(null);
  const [error, setError] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [selectedRequest, setSelectedRequest] = useState(null);
  const [actionType, setActionType] = useState(null);
  const [remarks, setRemarks] = useState('');
  const [successMessage, setSuccessMessage] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');

  const API_BASE = 'http://10.8.11.183:8000';
  const AUTH_TOKEN = 'TOKEN_admin';

  useEffect(() => {
    fetchData();
  }, [filter]);

  useEffect(() => {
    if (successMessage) {
      const timer = setTimeout(() => {
        setSuccessMessage(null);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [successMessage]);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [requestsRes, employeesRes] = await Promise.all([
        fetch(`${API_BASE}/api/attendance-request/all?status=${filter}&limit=100`, {
          headers: {
            'Authorization': `Bearer ${AUTH_TOKEN}`
          }
        }),
        fetch(`${API_BASE}/api/employees`, {
          headers: {
            'Authorization': `Bearer ${AUTH_TOKEN}`
          }
        })
      ]);

      if (!requestsRes.ok || !employeesRes.ok) {
        throw new Error('Failed to fetch data');
      }

      const requestsData = await requestsRes.json();
      const employeesData = await employeesRes.json();

      setRequests(requestsData.requests || []);
      
      const empMap = {};
      employeesData.employees?.forEach(emp => {
        empMap[emp.emp_id] = emp;
      });
      setEmployees(empMap);
    } catch (err) {
      setError('Failed to load data. Please check your connection and try again.');
      console.error('Error fetching data:', err);
    } finally {
      setLoading(false);
    }
  };

  const openApprovalModal = (request, type) => {
    setSelectedRequest(request);
    setActionType(type);
    setRemarks('');
    setShowModal(true);
  };

  const closeModal = () => {
    setShowModal(false);
    setSelectedRequest(null);
    setActionType(null);
    setRemarks('');
  };

  const handleApproval = async () => {
    if (!remarks.trim()) {
      setError('Please enter remarks before submitting');
      return;
    }

    setApproving(selectedRequest.id);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/attendance-request/${selectedRequest.id}/approve`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${AUTH_TOKEN}`
        },
        body: JSON.stringify({
          status: actionType,
          remarks: remarks.trim()
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to process request');
      }

      if (data.success) {
        setSuccessMessage(`Request ${actionType} successfully for ${data.name}`);
        closeModal();
        await fetchData();
      } else {
        setError(data.detail || 'Failed to process request');
      }
    } catch (err) {
      setError(err.message || 'Failed to process request. Please try again.');
      console.error('Error approving request:', err);
    } finally {
      setApproving(null);
    }
  };

  const getStatusBadge = (status) => {
    const styles = {
      pending: 'bg-amber-100 text-amber-800 ring-1 ring-amber-400/30',
      approved: 'bg-emerald-100 text-emerald-800 ring-1 ring-emerald-400/30',
      rejected: 'bg-rose-100 text-rose-800 ring-1 ring-rose-400/30'
    };

    const icons = {
      pending: <Clock className="w-4 h-4" />,
      approved: <CheckCircle className="w-4 h-4" />,
      rejected: <XCircle className="w-4 h-4" />
    };

    return (
      <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold ${styles[status]}`}>
        {icons[status]}
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </span>
    );
  };

  const getRequestTypeBadge = (type) => {
    const styles = {
      wfh: 'bg-indigo-100 text-indigo-800 ring-1 ring-indigo-400/30',
      manual_capture: 'bg-purple-100 text-purple-800 ring-1 ring-purple-400/30'
    };

    const labels = {
      wfh: 'WFH',
      manual_capture: 'Manual Entry'
    };

    return (
      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold ${styles[type]}`}>
        {labels[type] || type}
      </span>
    );
  };

  const filteredRequests = requests.filter(request => {
    if (!searchQuery) return true;
    const employee = employees[request.emp_id];
    const searchLower = searchQuery.toLowerCase();
    return (
      request.emp_id.toLowerCase().includes(searchLower) ||
      (employee?.name || '').toLowerCase().includes(searchLower) ||
      (request.name || '').toLowerCase().includes(searchLower) ||
      request.date.includes(searchQuery)
    );
  });

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600 font-medium">Loading requests...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-lg flex items-center justify-center">
                <Building2 className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Attendance Request Management</h1>
                <p className="text-sm text-gray-600 mt-0.5">Review and manage employee attendance requests</p>
              </div>
            </div>
            <button
              onClick={fetchData}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 font-medium"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>

          <div className="flex gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search by name, ID, or date..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>
            <div className="flex items-center gap-2 px-3 py-2 bg-white border border-gray-300 rounded-lg">
              <Filter className="w-4 h-4 text-gray-500" />
              <select
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                className="bg-transparent focus:outline-none text-sm font-medium text-gray-700 cursor-pointer"
              >
                <option value="all">All Status</option>
                <option value="pending">Pending</option>
                <option value="approved">Approved</option>
                <option value="rejected">Rejected</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 py-6">
        {error && (
          <div className="mb-4 bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
            <p className="text-sm text-red-800 flex-1">{error}</p>
            <button onClick={() => setError(null)} className="text-red-600 hover:text-red-800">
              <X className="w-5 h-5" />
            </button>
          </div>
        )}

        {successMessage && (
          <div className="mb-4 bg-emerald-50 border border-emerald-200 rounded-lg p-4 flex items-center gap-3">
            <CheckCircle className="w-5 h-5 text-emerald-600 flex-shrink-0" />
            <p className="text-sm text-emerald-800 flex-1">{successMessage}</p>
            <button onClick={() => setSuccessMessage(null)} className="text-emerald-600 hover:text-emerald-800">
              <X className="w-5 h-5" />
            </button>
          </div>
        )}

        {filteredRequests.length === 0 ? (
          <div className="bg-white rounded-lg border border-gray-200 p-12 text-center">
            <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">No requests found</h3>
            <p className="text-gray-600">
              {searchQuery 
                ? 'No requests match your search criteria.' 
                : `There are no ${filter !== 'all' ? filter : ''} attendance requests at the moment.`}
            </p>
          </div>
        ) : (
          <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-gray-50 border-b border-gray-200">
                    <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Employee</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Employee ID</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Date</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Type</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Time In</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Time Out</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Status</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {filteredRequests.map((request) => {
                    const employee = employees[request.emp_id];
                    return (
                      <tr key={request.id} className="hover:bg-gray-50 transition-colors">
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-3">
                            {employee?.image_url ? (
                              <img
                                src={employee.image_url}
                                alt={employee.name}
                                className="w-10 h-10 rounded-full object-cover"
                              />
                            ) : (
                              <div className="w-10 h-10 rounded-full bg-gray-100 flex items-center justify-center">
                                <User className="w-5 h-5 text-gray-400" />
                              </div>
                            )}
                            <div>
                              <p className="text-sm font-medium text-gray-900">
                                {employee?.name || request.name || 'Unknown'}
                              </p>
                            </div>
                          </div>
                        </td>

                        <td className="px-6 py-4">
                          <p className="text-sm text-gray-700 font-medium">{request.emp_id}</p>
                        </td>

                        <td className="px-6 py-4">
                          <div className="flex items-center gap-2">
                            <Calendar className="w-4 h-4 text-gray-400" />
                            <p className="text-sm text-gray-700">{request.date}</p>
                          </div>
                        </td>

                        <td className="px-6 py-4">
                          {getRequestTypeBadge(request.request_type)}
                        </td>

                        <td className="px-6 py-4">
                          <p className="text-sm text-gray-700 font-mono">{request.in_time || '—'}</p>
                        </td>

                        <td className="px-6 py-4">
                          <p className="text-sm text-gray-700 font-mono">{request.out_time || '—'}</p>
                        </td>

                        <td className="px-6 py-4">
                          {getStatusBadge(request.status)}
                        </td>

                        <td className="px-6 py-4">
                          {request.status === 'pending' ? (
                            <div className="flex items-center gap-2">
                              <button
                                onClick={() => openApprovalModal(request, 'approved')}
                                className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-emerald-600 text-white text-sm font-medium rounded-md hover:bg-emerald-700 transition-colors"
                              >
                                <CheckCircle className="w-4 h-4" />
                                Approve
                              </button>
                              <button
                                onClick={() => openApprovalModal(request, 'rejected')}
                                className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-rose-600 text-white text-sm font-medium rounded-md hover:bg-rose-700 transition-colors"
                              >
                                <XCircle className="w-4 h-4" />
                                Reject
                              </button>
                            </div>
                          ) : (
                            <button
                              onClick={() => {
                                setSelectedRequest(request);
                                setShowModal(true);
                                setActionType('view');
                              }}
                              className="text-sm text-indigo-600 hover:text-indigo-800 font-medium"
                            >
                              View Details
                            </button>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="px-6 py-3 bg-gray-50 border-t border-gray-200">
              <p className="text-sm text-gray-600">
                Showing <span className="font-semibold text-gray-900">{filteredRequests.length}</span> of{' '}
                <span className="font-semibold text-gray-900">{requests.length}</span> requests
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Modal */}
      {showModal && selectedRequest && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
              <h2 className="text-xl font-semibold text-gray-900">
                {actionType === 'view' ? 'Request Details' : actionType === 'approved' ? 'Approve Request' : 'Reject Request'}
              </h2>
              <button onClick={closeModal} className="text-gray-400 hover:text-gray-600 transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-6 space-y-6">
              {/* Employee Info */}
              <div className="flex items-center gap-4 p-4 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg border border-indigo-100">
                {employees[selectedRequest.emp_id]?.image_url ? (
                  <img
                    src={employees[selectedRequest.emp_id].image_url}
                    alt={employees[selectedRequest.emp_id].name}
                    className="w-16 h-16 rounded-full object-cover"
                  />
                ) : (
                  <div className="w-16 h-16 rounded-full bg-indigo-100 flex items-center justify-center">
                    <User className="w-8 h-8 text-indigo-600" />
                  </div>
                )}
                <div className="flex-1">
                  <p className="font-semibold text-gray-900 text-lg">
                    {employees[selectedRequest.emp_id]?.name || selectedRequest.name}
                  </p>
                  <p className="text-sm text-gray-600 mt-0.5">ID: {selectedRequest.emp_id}</p>
                  <div className="flex items-center gap-2 mt-2">
                    {getStatusBadge(selectedRequest.status)}
                    {getRequestTypeBadge(selectedRequest.request_type)}
                  </div>
                </div>
              </div>

              {/* Request Details Grid */}
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <p className="text-xs font-semibold text-gray-500 uppercase mb-1">Date</p>
                  <p className="text-sm font-medium text-gray-900">{selectedRequest.date}</p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <p className="text-xs font-semibold text-gray-500 uppercase mb-1">Request Type</p>
                  <p className="text-sm font-medium text-gray-900">
                    {selectedRequest.request_type === 'wfh' ? 'Work From Home' : 'Manual Entry'}
                  </p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <p className="text-xs font-semibold text-gray-500 uppercase mb-1">Time In</p>
                  <p className="text-sm font-medium text-gray-900 font-mono">{selectedRequest.in_time || 'N/A'}</p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <p className="text-xs font-semibold text-gray-500 uppercase mb-1">Time Out</p>
                  <p className="text-sm font-medium text-gray-900 font-mono">{selectedRequest.out_time || 'N/A'}</p>
                </div>
              </div>

              {/* Reason */}
              {selectedRequest.reason && (
                <div>
                  <p className="text-sm font-semibold text-gray-700 mb-2">Employee Reason:</p>
                  <div className="p-4 bg-indigo-50 rounded-lg border border-indigo-200">
                    <p className="text-sm text-gray-700">{selectedRequest.reason}</p>
                  </div>
                </div>
              )}

              {/* Admin Remarks */}
              {selectedRequest.remarks && selectedRequest.status !== 'pending' && (
                <div>
                  <p className="text-sm font-semibold text-gray-700 mb-2">Admin Remarks:</p>
                  <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                    <p className="text-sm text-gray-700">{selectedRequest.remarks}</p>
                  </div>
                </div>
              )}

              {/* Action Area */}
              {actionType !== 'view' && selectedRequest.status === 'pending' && (
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Remarks <span className="text-red-500">*</span>
                  </label>
                  <textarea
                    value={remarks}
                    onChange={(e) => setRemarks(e.target.value)}
                    placeholder={`Enter your remarks for ${actionType === 'approved' ? 'approval' : 'rejection'}...`}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none"
                    rows="4"
                  />
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex gap-3 pt-4 border-t border-gray-200">
                {actionType === 'view' || selectedRequest.status !== 'pending' ? (
                  <button
                    onClick={closeModal}
                    className="flex-1 px-4 py-2.5 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors font-medium"
                  >
                    Close
                  </button>
                ) : (
                  <>
                    <button
                      onClick={handleApproval}
                      disabled={approving || !remarks.trim()}
                      className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-white font-medium transition-colors ${
                        actionType === 'approved'
                          ? 'bg-emerald-600 hover:bg-emerald-700'
                          : 'bg-rose-600 hover:bg-rose-700'
                      } disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                      {approving ? (
                        <>
                          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                          Processing...
                        </>
                      ) : (
                        <>
                          {actionType === 'approved' ? <CheckCircle className="w-4 h-4" /> : <XCircle className="w-4 h-4" />}
                          {actionType === 'approved' ? 'Approve' : 'Reject'}
                        </>
                      )}
                    </button>
                    <button
                      onClick={closeModal}
                      disabled={approving}
                      className="px-6 py-2.5 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Cancel
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AttendanceRequestPage;